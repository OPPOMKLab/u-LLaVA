"""
Copyright 2023 OPPO

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import argparse
import transformers
from typing import Optional
from tasks import setup_task
from utils.tools import datetime_print
from utils.config_builder import Config
from dataclasses import dataclass, field
from models import UllavaCoreForCausalLM, UllavaCoreConfig
from transformers import Trainer, LlamaTokenizer, CLIPVisionConfig, CLIPVisionModel, AutoConfig
from models.tools import smart_special_token_and_embedding_resize, multi_modal_resize_token_embedding
from models import DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_UNK_TOKEN, DEFAULT_PAD_TOKEN, \
    DEFAULT_IMG_PATCH_TOKEN, DEFAULT_IMG_START_TOKEN, DEFAULT_IMG_END_TOKEN, DEFAULT_VID_PATCH_TOKEN,\
    DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    disable_tqdm: bool = field(default=False)
    report_to: str = field(default='none')


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train(config):
    model_args, dataset_args, eval_dataset_args, training_args, task_args, processor_args = config.assign_config()

    train_parser = transformers.HfArgumentParser(TrainingArguments)
    training_args = train_parser.parse_dict(training_args)[0]

    datetime_print('Loading Tokenizer')
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.llm_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    base_config = AutoConfig.from_pretrained(model_args.llm_path)
    if base_config.model_type == 'llama':
        # for llama/llama2/vucuna
        vision_config = CLIPVisionConfig.from_pretrained(model_args.vision_encoder)
        model_config = UllavaCoreConfig(vision_config=vision_config.to_dict(),
                                        vision_hidden_layer=model_args.vision_hidden_layer,
                                        projector_from_scratch=model_args.projector_from_scratch,
                                        mm_token_ids=None,
                                        **base_config.to_dict())
    elif base_config.model_type == 'ullava_core':
        base_config.projector_from_scratch = model_args.projector_from_scratch
        model_config = UllavaCoreConfig(**base_config.to_dict())
    else:
        print('Unknown model type')
        raise NotImplementedError

    dtype = 'fp16' if training_args.fp16 else 'bf16' if training_args.bf16 else 'fp32'
    torch_dtype = torch.float16 if dtype == 'fp16' else torch.bfloat16 if dtype == 'bf16' else torch.float32

    datetime_print('Initializing uLLaVA Core')
    model = UllavaCoreForCausalLM.from_pretrained(
        model_args.llm_path,
        config=model_config,
        torch_dtype=torch_dtype,
        cache_dir=training_args.cache_dir
    )
    # `use_cache=True` is incompatible with gradient checkpointing
    model.config.use_cache = False

    model_vocab_size = model.get_output_embeddings().weight.size(0)

    if tokenizer.pad_token is None:
        smart_special_token_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    # add special tokens, can be disable
    tokenizer.add_special_tokens({
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    })

    if base_config.model_type == 'llama':
        # LLaMA model, addvisual tokens, load vision model
        datetime_print('LLaMA model, Loading CLIP Vision Encoder')
        model.vision_encoder = CLIPVisionModel.from_pretrained(
            model_args.vision_encoder,
            torch_dtype=torch_dtype,
            cache_dir=training_args.cache_dir,
        )

        mm_tokens = {
            'IMG_PATCH': DEFAULT_IMG_PATCH_TOKEN, 'VID_PATCH': DEFAULT_VID_PATCH_TOKEN,
            'IMG_START': DEFAULT_IMG_START_TOKEN, 'IMG_END': DEFAULT_IMG_END_TOKEN,
            'VID_START': DEFAULT_VID_START_TOKEN, 'VID_END': DEFAULT_VID_END_TOKEN
        }

        multi_modal_resize_token_embedding(
            mm_tokens=mm_tokens,
            tokenizer=tokenizer,
            model=model
            )

        model.init_mm_tokens(tokenizer, mm_tokens)

    num_new_tokens = len(tokenizer) - model_vocab_size
    datetime_print('Number of newly added tokens: {0}'.format(num_new_tokens))

    if model_args.projector_from_scratch:
        datetime_print('Pre-training stage')
        # According to LLaVA, freeze all except projector and input embeddings during pre-training
        model.requires_grad_(False)
        model.vision_encoder.requires_grad_(False)

        for p in model.vision_projector.parameters():
            p.requires_grad = True
        for p in model.get_input_embeddings().parameters():
            p.requires_grad = True
        for p in model.get_output_embeddings().parameters():
            p.requires_grad = False
    else:
        datetime_print('Fine-tuning Stage')
        model.vision_encoder.requires_grad_(False)

    task = setup_task(task_args)
    processor_dict = task.build_processors(processor_args)
    data_collator = task.build_collator(tokenizer.pad_token_id)
    train_dataset = task.build_datasets(dataset_args, tokenizer, processor_dict, conv_type=model_args.conv_type)

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=train_dataset,
                      data_collator=data_collator)

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--cfg_path", 
        default='./configs/train/ullava_core_stage1.yaml', 
        help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    cfg = Config(parser.parse_args().cfg_path)

    cfg.pretty_print()

    train(cfg)
