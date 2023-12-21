"""
u-LLaVA inference
Partially Adapted form: https://github.com/dvlab-research/LISA/blob/main/chat.py
"""

import os
import sys
sys.path.append(os.getcwd())
import re
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from peft import PeftModel
import torch.nn.functional as F
from utils.tools import load_image
from transformers import LlamaTokenizer
import torchvision.transforms as transforms
from torchvision.utils import draw_bounding_boxes
from datasets.processors.clip_processor import CLIPProcessor
from torchvision.ops import masks_to_boxes, box_convert, box_iou
from utils.conversation import default_conversation, SeparatorStyle
from models.segment_anything.utils.transforms import ResizeLongestSide
from models import UllavaForCausalLM, GroundingModule, KeywordsStoppingCriteria, \
    DEFAULT_IMG_END_TOKEN, DEFAULT_IMG_START_TOKEN, DEFAULT_IMG_TOKEN, DEFAULT_IMG_PATCH_TOKEN,\
    DEFAULT_TAG_START, DEFAULT_TAG_END


def preprocess(
        x,
        pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
        pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
        img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def select_bbox(dino_boxes, pred_mask, image_pil):
    w, h = image_pil.size
    dino_boxes = dino_boxes * torch.Tensor([w, h, w, h])
    dino_boxes = box_convert(boxes=dino_boxes, in_fmt="cxcywh", out_fmt="xyxy")

    # pred_mask > 0 after seg
    pred_mask = torch.tensor(pred_mask).unsqueeze(0)
    box_temp = masks_to_boxes(pred_mask)
    box_sam = box_temp[0]

    if len(dino_boxes) > 0:
        box = box_sam.expand(dino_boxes.size())
        ious = box_iou(box, dino_boxes)
        ious = torch.einsum('i i -> i', ious)  # take diag elem
        max_idx = torch.argmax(ious)
    else:
        max_idx = -1

    return max_idx, box_temp


def main(args):
    os.makedirs(args.vis_save_path, exist_ok=True)

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

    clip_image_processor = CLIPProcessor(args.clip_path)
    transform = ResizeLongestSide(args.image_size)

    grounding_dino = GroundingModule(args.grounding_path, device='cuda')

    model.eval()
    grounding_dino.eval()

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
        image_np = np.array(image_pil)

        raw_size_list = [image_np.shape[:2]]

        image_clip = clip_image_processor(image_np).unsqueeze(0).cuda().to(dtype)

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda().to(dtype)

        input_ids = tokenizer(prompt).input_ids
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        output_ids, pred_masks = model.evaluate(
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
        print('uLLaVA Core:', text_output)

        pattern = re.escape(DEFAULT_TAG_START) + '(.*?)' + re.escape(DEFAULT_TAG_END)
        tags = re.findall(pattern, text_output)
        grounding_tags = ','.join(tags)

        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            save_path = "{}/{}_mask_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            cv2.imwrite(save_path, pred_mask * 255)
            print("{} has been saved.".format(save_path))

            save_path = "{}/{}_masked_img_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            save_img = image_np.copy()
            save_img[pred_mask] = (
                    image_np * 0.5
                    + pred_mask[:, :, None].astype(np.uint8) * np.array([133, 131, 230]) * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))

            pred_boxes, logits, phrases = grounding_dino.prompt2boxes(image_pil, grounding_tags)
            
            try:
                max_idx, mask2box = select_bbox(pred_boxes, pred_mask, image_pil)
                save_img = image_np.copy()
                if max_idx >= 0:
                    annotated_frame = grounding_dino.annotate(save_img, pred_boxes, logits, phrases, box_id=max_idx)
                    save_path = "{}/{}_gd_box_{}.jpg".format(
                        args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
                    )
                    cv2.imwrite(save_path, annotated_frame)
    
                image_ts = (transforms.ToTensor()(save_img) * 255).to(torch.uint8)
                drawn_box = draw_bounding_boxes(image_ts, mask2box, colors="orange", width=3)
                img_box = transforms.functional.to_pil_image(drawn_box, 'RGB')
                save_path = "{}/{}_m2b_box_{}.jpg".format(
                    args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
                )
                img_box.save(save_path)
                print("Grounding and Mask2box has been saved.")
            except:
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="uLLaVA Segmentation and Grounding chat")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--llm_path",
                        default="./exp/ullava/stage2_lora")
    parser.add_argument("--grounding_path",
                        default="./model_zoo/grounding_dino/groundingdino_swint_ogc.pth")
    parser.add_argument(
        "--clip_path", default="./model_zoo/clip-vit-large-patch14",
        type=str
    )
    parser.add_argument("--vis_save_path", default="./temp", type=str)
    parser.add_argument(
        "--dtype",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--conv_type",
        default="conv_sep2",
        type=str,
        choices=["conv_simple", "conv_sep2", "conv_llama2"],
    )

    cfg = parser.parse_args()
    main(cfg)
