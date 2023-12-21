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
from transformers import AutoConfig
from utils.tools import datetime_print
from utils.config_builder import Config
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model
from metrics.meter import AverageMeter, Summary
from trainers.ullava_trainer import SegmentationTrainer
from models import UllavaCoreForCausalLM, UllavaForCausalLM, UllavaConfig, \
    DEFAULT_SEG_TOKEN, DEFAULT_TAG_START, DEFAULT_TAG_END


@dataclass
class ModelArguments:
    arch: str = field(default='ullava')
    llm_path: str = field(default='./path_to_ullava_core')
    sam_pretrained: str = field(default='./path_to_sam/sam_vit_h_4b8939.pth')
    lora_r: int = field(default=-1)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default='q_proj,v_proj')
    train_mask_decoder: bool = field(default=True)
    out_dim: int = field(default=256)
    ce_loss_weight: float = field(default=1.0)
    dice_loss_weight: float = field(default=0.5)
    bce_loss_weight: float = field(default=2.0)
    conv_type: str = field(default='conv_sep2')
    image_size: int = field(default=1024)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    include_inputs_for_metrics: bool = field(default=True)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    disable_tqdm: bool = field(default=False)
    report_to: str = field(default='none')


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, is_peft: bool):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()

    if trainer.args.should_save:
        if is_peft:
            cpu_state_dict = {key.replace('.base_model.model', ''): value.cpu() for key, value in
                              state_dict.items() if 'lora_' not in key}

            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        else:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}

        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def find_linear_layers(model, lora_target_modules):
    """
    Borrowed from https://github.com/dvlab-research/LISA/blob/main/train_ds.py-find_linear_layers
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        # print(name)
        if (
                isinstance(module, cls)
                and all(
            [
                x not in name
                for x in [
                "visual_model",
                "vision_encoder",
                "vision_projector",
                "text_hidden_fcs",
            ]
            ]
        )
                and any([x in name for x in lora_target_modules])
        ):
            lora_module_names.add(name)
    return sorted(list(lora_module_names))


def compute_metrics(eval_output):
    # print(eval_output)
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    results = eval_output.predictions

    for res in results:
        intersection, union, acc_iou = res[:2], res[2:4], res[4:]
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=1)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    return {'cIoU': ciou, 'gIoU': giou}


def train(config):
    model_args, dataset_args, eval_dataset_args, training_args, task_args, processor_args = config.assign_config()

    model_parser = transformers.HfArgumentParser(ModelArguments)
    model_args = model_parser.parse_dict(model_args)[0]

    train_parser = transformers.HfArgumentParser(TrainingArguments)
    training_args = train_parser.parse_dict(training_args)[0]

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.llm_path,
        cache_dir=None,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens([DEFAULT_SEG_TOKEN, DEFAULT_TAG_START, DEFAULT_TAG_END], special_tokens=True)
    model_args.seg_token_idx = tokenizer.convert_tokens_to_ids(DEFAULT_SEG_TOKEN)

    base_config = AutoConfig.from_pretrained(model_args.llm_path)
    if base_config.model_type == 'ullava_core':
        base_config.projector_from_scratch = False  # False or True result in similar performance
        model_config = UllavaConfig(llm_config=base_config.to_dict(),
                                    train_mask_decoder=model_args.train_mask_decoder,
                                    out_dim=model_args.out_dim,
                                    ce_weight=model_args.ce_loss_weight,
                                    bce_weight=model_args.bce_loss_weight,
                                    dice_weight=model_args.dice_loss_weight,
                                    sep_token_idx=model_args.seg_token_idx)
    elif base_config.model_type == 'ullava':
        model_config = UllavaConfig(**base_config.to_dict())
    else:
        print('Unknown model type')
        raise NotImplementedError

    dtype = 'fp16' if training_args.fp16 else 'bf16' if training_args.bf16 else 'fp32'
    torch_dtype = torch.float16 if dtype == 'fp16' else torch.bfloat16 if dtype == 'bf16' else torch.float32

    if base_config.model_type == 'ullava_core':
        # from config
        datetime_print('Building uLLaVA from modules')
        model = UllavaForCausalLM(config=model_config)
        datetime_print('Step1: Building uLLaVA Core')
        model.llm = UllavaCoreForCausalLM.from_pretrained(
            model_args.llm_path,
            torch_dtype=torch_dtype,
            cache_dir=training_args.cache_dir,
        )
        datetime_print('Step2: Loading Segmentation checkpoint')
        model.load_visual_checkpoint(model_args.sam_pretrained)
    elif base_config.model_type == 'ullava':
        datetime_print('Building uLLaVA from pretrained')
        model = UllavaForCausalLM.from_pretrained(
            model_args.llm_path,
            config=model_config,
            torch_dtype=torch_dtype,
            cache_dir=training_args.cache_dir
        )
    else:
        raise NotImplementedError

    for p in model.llm.vision_encoder.parameters():
        p.requires_grad = False
    for p in model.llm.vision_projector.parameters():
        p.requires_grad = False

    model.llm.resize_token_embeddings(len(tokenizer))
    # make sure save the right config after resize embeddings
    model.config.llm_config = model.llm.config

    model.llm.enable_input_require_grads()
    model.llm.gradient_checkpointing_enable()

    lora_r = model_args.lora_r
    if lora_r > 0:
        lora_alpha = model_args.lora_alpha
        lora_dropout = model_args.lora_dropout
        lora_target_modules = find_linear_layers(
            model.llm, model_args.lora_target_modules.split(",")
        )

        print(lora_target_modules)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.llm = get_peft_model(model.llm, lora_config)
        model.llm.print_trainable_parameters()
    else:
        for p in model.llm.model.parameters():
            p.requires_grad = True
        for p in model.llm.lm_head.parameters():
            p.requires_grad = True
        for p in model.llm.vision_projector.parameters():
            p.requires_grad = True

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
                [
                    x in n
                    for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
                ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    task = setup_task(task_args)
    processor_dict = task.build_processors(processor_args)
    data_collator = task.build_collator(tokenizer.pad_token_id)
    train_dataset = task.build_datasets(dataset_args, tokenizer, processor_dict, conv_type=model_args.conv_type)

    if eval_dataset_args is not None:
        eval_dataset = task.build_datasets(eval_dataset_args, tokenizer, processor_dict, conv_type=model_args.conv_type)
    else:
        eval_dataset = None

    trainer = SegmentationTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_state()

    if lora_r > 0:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, is_peft=True)
        model.llm.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, is_peft=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument(
        "--cfg_path", 
        default='./configs/train/ullava_stage2_lora.yaml', 
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

