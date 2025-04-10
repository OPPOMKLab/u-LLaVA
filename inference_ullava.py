"""
u-LLaVA inference
Partially Adapted form: https://github.com/dvlab-research/LISA/blob/main/chat.py
"""

import os
import cv2
import torch
import argparse
import numpy as np
import torchvision
from peft import PeftModel
from utils.tools import load_image
from transformers import LlamaTokenizer
from torchvision.utils import draw_bounding_boxes
from dataset.processors.clip_processor import CLIPProcessor
from dataset.tools.mask_toolbox import SegToolBox, DetToolBox
from utils.conversation import default_conversation, SeparatorStyle
from models import UllavaForCausalLM, KeywordsStoppingCriteria, \
    DEFAULT_IMG_END_TOKEN, DEFAULT_IMG_START_TOKEN, DEFAULT_IMG_TOKEN, DEFAULT_IMG_PATCH_TOKEN


def main(args):
    os.makedirs(args.result_dir, exist_ok=True)

    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    # Create model
    tokenizer = LlamaTokenizer.from_pretrained(
        args.llm_path,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=False
    )

    model = UllavaForCausalLM.from_pretrained(
        args.llm_path,
        torch_dtype=dtype,
    )

    if args.lora_r > 0:
        model.llm = PeftModel.from_pretrained(model.llm, args.llm_path, torch_dtype=dtype)

    model.cuda()

    clip_image_processor = CLIPProcessor(args.clip_path, args.aspect_ratio)
    seg_tool, det_tool = SegToolBox(), DetToolBox()

    model.eval()

    while True:
        conv = default_conversation.copy()
        conv.messages = []

        prompt = input("Please input your prompt: ")
        prompt = DEFAULT_IMG_TOKEN + "\n" + prompt

        replace_token = (
                DEFAULT_IMG_START_TOKEN + DEFAULT_IMG_PATCH_TOKEN * 256 + DEFAULT_IMG_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMG_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        # print(prompt)

        image_path = input("Please input the image path: ")
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        image_pil = load_image(image_path)
        width, height = image_pil.size
        image_np = np.array(image_pil)

        raw_size_list = [image_np.shape[:2]]

        image_clip = clip_image_processor(image_np).unsqueeze(0).cuda().to(dtype)

        image = seg_tool.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = seg_tool.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda().to(dtype)

        input_ids = tokenizer(prompt).input_ids
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        output_ids, pred_masks, pred_boxes = model.evaluate(
            image,
            image_clip,
            input_ids,
            raw_size_list,
            resize_list,
            max_new_tokens=512,
            stopping_criteria=[stopping_criteria]
        )
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        text_output = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        print('uLLaVA:', text_output)

        for i, (pred_mask, pred_bbox) in enumerate(zip(pred_masks, pred_boxes)):
            if pred_mask.shape[0] == 0 or pred_bbox.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            pred_bbox = pred_bbox[0].detach().cpu()
            pred_bbox = torch.tensor(det_tool.denormalize_padded_xyxy(pred_bbox, width, height)).unsqueeze(0)

            save_path = "{}/{}_mask_{}.jpg".format(
                args.result_dir, image_path.split("/")[-1].split(".")[0], i
            )
            cv2.imwrite(save_path, pred_mask * 255)
            print("{} has been saved.".format(save_path))

            save_path = "{}/{}_masked_img_{}.jpg".format(
                args.result_dir, image_path.split("/")[-1].split(".")[0], i
            )
            save_img = image_np.copy()
            save_img[pred_mask] = (
                    image_np * 0.5
                    + pred_mask[:, :, None].astype(np.uint8) * np.array([133, 131, 230]) * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))

            save_img = image_np.copy()
            image_ts = (torchvision.transforms.ToTensor()(save_img) * 255).to(torch.uint8)
            drawn_box = draw_bounding_boxes(image_ts, pred_bbox, colors='orange', width=3)
            img_box = torchvision.transforms.functional.to_pil_image(drawn_box, 'RGB')
            save_path = "{}/{}_bbox_{}.jpg".format(
                args.result_dir, image_path.split("/")[-1].split(".")[0], i
            )
            img_box.save(save_path)
            print("{} has been saved.".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="uLLaVA Segmentation and Grounding chat")
    parser.add_argument("--lora_r", default=-1, type=int)
    parser.add_argument("--llm_path",
                        default="./exp/ullava")
    parser.add_argument(
        "--clip_path", default="./model_zoo/clip-vit-large-patch14",
        type=str
    )
    parser.add_argument(
        "--aspect_ratio", default="pad",
        type=str
    )
    parser.add_argument("--result_dir", default="./temp", type=str)
    parser.add_argument(
        "--dtype",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--conv_type",
        default="conv_sep2",
        type=str,
        choices=["conv_simple", "conv_sep2", "conv_llama2"],
    )

    cfg = parser.parse_args()
    main(cfg)
