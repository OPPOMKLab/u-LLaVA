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
import cv2
import copy
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from models.segment_anything.utils.transforms import ResizeLongestSide
from datasets.datasets.base_dataset import BaseDataset, preprocess, preprocess_image_text
from models import DEFAULT_SEG_TOKEN, DEFAULT_IMG_TOKEN, DEFAULT_TAG_START, DEFAULT_TAG_END


CLASS_TOKEN = '<class>'
TAG_TOKEN = '<tag>'


class SalientSegDataset(BaseDataset):
    class_map = {}

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
        super().__init__(
            vis_processor=vis_processor,
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
        self.aspect_ratio = self.vis_processor.aspect_ratio

        self.num_sentence_per_item = 1
        # FOR SAM
        self.sam_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.sam_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.sam_size = 1024
        self.ignore_label = 255
        self.sam_transform = ResizeLongestSide(self.sam_size)

    def preprocess_for_sam(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.sam_mean) / self.sam_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.sam_size - h
        padw = self.sam_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    @staticmethod
    def get_label(label_path):
        # MSRA-10K, MSRA-B: bg - 0, gt - 255 (white)
        label = Image.open(label_path)
        label = np.array(label)

        return label

    def build_sample(self, index):
        item = self.annotation[index]
        image_path = os.path.join(self.vis_root, item['image_path'])
        label_path = os.path.join(self.vis_root, item['label_path'])

        # single image, multi-round
        caption = item['caption']

        conversations = []
        roles = ['human', 'gpt']

        question = self.random_choice_template()

        conversations.append({
            'from': roles[0],
            'value': question,
        })
        conversations.append({
            'from': roles[1],
            'value': f'It is {caption.lower()}. Info: {DEFAULT_SEG_TOKEN}; '
                     f'{DEFAULT_TAG_START}{caption.lower()}{DEFAULT_TAG_END}.'
        })

        sample = {
            'image_path': image_path,
            'target': {
                'label_path': label_path,
            },
            'conversations': conversations
        }

        return sample

    def __getitem__(self, idx):
        sample = self.build_sample(idx)
        # image = Image.open(sample['image_path']).convert('RGB')
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        conversation_list = sample['conversations']
        label_path = sample["target"]["label_path"]

        label = self.get_label(label_path)

        # preprocess images for clip
        image_clip = self.vis_processor(image)
        image_sam = self.sam_transform.apply_image(image)  # preprocess images for sam
        resize = image_sam.shape[:2]

        # image_sam = self.preprocess_for_sam(pil_to_tensor(image_sam))
        image_sam = self.preprocess_for_sam(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())

        sources = preprocess_image_text(copy.deepcopy(conversation_list),
                                        cur_token_len=self.image_token_len)

        data_dict = preprocess(sources, self.tokenizer, self.conv_type)
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])

        # For Segmentation
        seg_label = torch.from_numpy(label).long()
        masks = [seg_label == 255]

        seg_mask = torch.stack(masks, dim=0).float()
        raw_size = [seg_mask.shape[1], seg_mask.shape[2]]

        image_dict = {'image': image_clip, 'image_sam': image_sam,
                      'seg_mask': seg_mask, 'raw_size': raw_size, 'resize': resize}

        data_dict.update(image_dict)

        return data_dict


class ValSalientSegDataset(SalientSegDataset):
    @staticmethod
    def random_choice_template():
        # no need of templates for validation
        return DEFAULT_IMG_TOKEN + '\n' + f"Find the salient object in the image."
