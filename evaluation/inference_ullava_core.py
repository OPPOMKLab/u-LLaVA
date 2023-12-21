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

import os
import sys
sys.path.append(os.getcwd())
import torch
import argparse
from utils.tools import load_image
from transformers import LlamaTokenizer
from utils.tools import disable_torch_init
from datasets.processors.clip_processor import CLIPProcessor
from utils.conversation import conversation_lib, SeparatorStyle
from models import UllavaCoreForCausalLM, KeywordsStoppingCriteria, \
    DEFAULT_IMG_PATCH_TOKEN, DEFAULT_IMG_START_TOKEN, DEFAULT_IMG_END_TOKEN

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False


def eval_model(args):
    disable_torch_init()
    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    model = UllavaCoreForCausalLM.from_pretrained(
        args.model_name,
        use_cache=True,
        torch_dtype=dtype
    ).cuda()

    tokenizer = LlamaTokenizer.from_pretrained(args.model_name, use_fast=False)
    image_processor = CLIPProcessor(args.clip_path, aspect_ratio=None)

    vision_config = model.config.vision_config
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    while True:
        conv = conversation_lib[args.conv_mode].copy()
        conv.messages = []
        qs = input("Please input your prompt: ")
        qs = DEFAULT_IMG_START_TOKEN + DEFAULT_IMG_PATCH_TOKEN * image_token_len + DEFAULT_IMG_END_TOKEN + '\n' + qs
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        image_path = input("Please input the image path: ")
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        image = load_image(image_path)
        image_tensor = image_processor(image)
        image_tensor = image_tensor.unsqueeze(0).cuda().to(dtype)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria]
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print('uLLaVA Core:', outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="./exp/ullava/stage1")
    parser.add_argument("--clip_path", type=str,
                        default='./model_zoo/clip-vit-large-patch14')
    parser.add_argument("--conv_mode", type=str, default='conv_simple')
    parser.add_argument(
        "--dtype",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    cfg = parser.parse_args()

    eval_model(cfg)

