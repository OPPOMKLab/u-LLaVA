"""
Base dataset class of uLLaVA
Partially Adapted form: https://github.com/dvlab-research/LISA/blob/main/utils/refer_seg_dataset.py
"""

import os
import cv2
import copy
import torch
import numpy as np
from pycocotools import mask
from dataset.datasets.base_dataset import BaseDataset, preprocess, preprocess_image_text
from models import DEFAULT_SEG_TOKEN, DEFAULT_LOC_TOKEN, DEFAULT_IMG_TOKEN, DEFAULT_TAG_START, DEFAULT_TAG_END


CLASS_TOKEN = '<class>'


class ResDataset(BaseDataset):
    num_sentence_per_item = 3

    def __init__(self,
                 vis_processor,
                 tokenizer,
                 vis_root,
                 ann_root,
                 template_root,
                 portion=1,
                 image_token_len=256,
                 seed=42,
                 data_type='image',
                 conv_type='conv_simple'
                 ):
        """
        vis_root (string):
        ann_root (string): jsonl file path
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor,
                         tokenizer=tokenizer,
                         vis_root=vis_root,
                         ann_root=ann_root,
                         template_root=template_root,
                         seed=seed,
                         portion=portion,
                         data_type=data_type,
                         conv_type=conv_type
                         )

        self.image_token_len = image_token_len

    def build_sample(self, index):
        item = self.annotation[index]
        segmentation = item['segmentation']
        category = item['category']
        bbox = item['bbox']
        image_path = os.path.join(self.vis_root, item['image_path'])

        # single image, multi-rounds
        sentences = item['sentences']
        if len(sentences) > self.num_sentence_per_item:
            sentences = np.random.choice(sentences, self.num_sentence_per_item, replace=False)

        conversations = []
        roles = ['human', 'gpt']
        for idx, sentence in enumerate(sentences):
            question = self.random_choice_template().replace(CLASS_TOKEN, sentence)
            if idx != 0:
                question = question.replace(DEFAULT_IMG_TOKEN, '')

            conversations.append({
                'from': roles[0],
                'value': question,
            })
            conversations.append({
                'from': roles[1],
                'value': f'Sure. Mask: {DEFAULT_SEG_TOKEN}; Location: {DEFAULT_LOC_TOKEN}; '
                         f'{DEFAULT_TAG_START}{category.lower()}{DEFAULT_TAG_END}.'
            })

        sample = {
            'image_path': image_path,
            'target': {
                'segmentation': segmentation,
                'bbox': bbox,
                'height': item['height'],
                'width': item['width']
            },
            'conversations': conversations
        }

        return sample

    def __getitem__(self, idx):
        sample = self.build_sample(idx)
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        conversation_list = sample['conversations']
        segmentation = sample['target']['segmentation']
        bbox = sample['target']['bbox']
        height, width = sample['target']['height'], sample['target']['width']

        # preprocess images for clip
        image_clip = self.vis_processor(image)
        # preprocess images for sam
        image_sam = self.seg_tool.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.seg_tool.preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())

        sources = preprocess_image_text(copy.deepcopy(conversation_list),
                                        cur_token_len=self.image_token_len)

        data_dict = preprocess(sources, self.tokenizer, self.conv_type)
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])

        # For Segmentation
        masks, boxes = [], []
        for _ in range(len(conversation_list) // 2):
            if len(segmentation) == 0:
                # non mask cases
                m = np.zeros((height, width)).astype(np.uint8)
                masks.append(m)

            if type(segmentation[0]) == list:  # polygon
                rle = mask.frPyObjects(
                    segmentation, height, width
                )
            else:
                rle = segmentation
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            # sometimes there are multiple binary map (corresponding to multiple seg)
            m = np.sum(m, axis=2)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)

            # convert [x, y, w, h] to [x1, y1, x2, y2]
            xyxy = self.det_tool.xywh2xyxy(bbox)
            # pad and normalize bbox
            normalized_bbox = self.det_tool.pad_normalize_xyxy(xyxy, width, height)
            boxes.append(normalized_bbox)

        masks = np.stack(masks, axis=0)
        seg_mask = torch.from_numpy(masks).float()
        raw_size = [seg_mask.shape[1], seg_mask.shape[2]]

        boxes = np.stack(boxes, axis=0)
        boxes = torch.from_numpy(boxes).float()

        image_dict = {'image': image_clip, 'image_sam': image_sam, 'seg_mask': seg_mask,
                      'boxes': boxes, 'raw_size': raw_size, 'resize': resize, }

        data_dict.update(image_dict)

        return data_dict


class ValResDataset(ResDataset):
    num_sentence_per_item = 10

    def random_choice_template(self):
        return DEFAULT_IMG_TOKEN + '\n' + f"Output the segmentation mask of the {CLASS_TOKEN} in the image."
