import os
import torch
import numpy as np
from transformers import LlamaTokenizer
from utils.conversation import SeparatorStyle
from dataset.processors.clip_processor import CLIPProcessor
from dataset.tools.mask_toolbox import DetToolBox, SegToolBox
from models import UllavaForCausalLM, KeywordsStoppingCriteria, DEFAULT_IMG_END_TOKEN,\
    DEFAULT_IMG_START_TOKEN, DEFAULT_IMG_TOKEN, DEFAULT_IMG_PATCH_TOKEN


class Chat:
    def __init__(self, args):
        self.args = args
        self.dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16 if args.dtype == 'bf16' else torch.float32

        self.tokenizer = LlamaTokenizer.from_pretrained(
            args.llm_path,
            cache_dir=None,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=False,
            legacy=False
        )

        self.model = UllavaForCausalLM.from_pretrained(
            args.llm_path,
            torch_dtype=self.dtype,
        )

        self.model = self.model.cuda()
        self.seg_tool = SegToolBox()
        self.det_tool = DetToolBox()

        self.clip_image_processor = CLIPProcessor(args.clip_processor, args.aspect_ratio)
        self.model_max_length = args.model_max_length

    def seg(self, user_message, pil_img, conv):
        pil_img = pil_img.convert('RGB')
        prompt = user_message
        prompt = DEFAULT_IMG_TOKEN + " " + prompt
        image_token_len = 256
        replace_token = DEFAULT_IMG_PATCH_TOKEN * image_token_len
        replace_token = DEFAULT_IMG_START_TOKEN + replace_token + DEFAULT_IMG_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMG_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_np = np.array(pil_img)
        raw_size_list = [image_np.shape[:2]]
        image_clip = self.clip_image_processor(image_np).unsqueeze(0).cuda().to(self.dtype)

        image = self.seg_tool.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = self.seg_tool.preprocess(
            torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda().to(self.dtype)

        input_ids = self.tokenizer(prompt).input_ids
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        output_ids, pred_masks, pred_boxes = self.model.evaluate(
            image,
            image_clip,
            input_ids,
            raw_size_list,
            resize_list,
            max_new_tokens=self.model_max_length,
            stopping_criteria=[stopping_criteria]
        )
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        text_output = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        return text_output, pred_masks, pred_boxes

