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
import random
from PIL import Image
import torch.nn.functional as F
from dataset.datasets.base_dataset import BaseDataset, preprocess, preprocess_image_text


class LLaVADataset(BaseDataset):
    def __init__(self,
                 vis_processor,
                 tokenizer,
                 vis_root,
                 ann_root,
                 portion=1,
                 image_token_len=256,
                 data_type='image',
                 conv_type='conv_simple'
                 ):
        """
        vis_root (string): Root directory of Llava images (e.g. webvid_eval/video/)
        ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor,
                         tokenizer=tokenizer,
                         vis_root=vis_root,
                         ann_root=ann_root,
                         portion=portion,
                         data_type=data_type,
                         conv_type=conv_type
                         )

        self.image_token_len = image_token_len

    def __getitem__(self, index):
        """
        "image_id" is kept to stay compatible with the COCO evaluation format
        some datasets contain mixed sources, such as SQA (w/, w/o image)
        :param index:
        :return:
        """
        num_retries = 10  # skip error images
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]
                conversation_list = sample['conversations']

                if 'image' in sample:
                    image_path = os.path.join(self.vis_root, sample['image'])
                    image = Image.open(image_path).convert("RGB")

                    # process image for clip
                    image = self.vis_processor(image)

                    sources = preprocess_image_text(copy.deepcopy(conversation_list),
                                                    cur_token_len=self.image_token_len)
                else:
                    image = None
                    sources = [copy.deepcopy(conversation_list)]

                data_dict = preprocess(sources, self.tokenizer, self.conv_type)
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                 labels=data_dict["labels"][0])

                # image exist: process by CLIP, non image: no 'image' dict
                if image is not None:
                    data_dict['image'] = image

            except Exception as error:
                image_path = self.annotation[index]['image'] if 'image' in self.annotation[index] else str(index)
                print(f"Failed to load example {image_path}, Error: {error}. "
                      f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        return data_dict


class LLaVASegDataset(LLaVADataset):
    def __getitem__(self, index):
        num_retries = 10  # skip error images
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]
                conversation_list = sample['conversations']

                # has image
                image_path = os.path.join(self.vis_root, sample['image'])
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                raw_size = image.shape[:2]

                # preprocess images for clip
                image_clip = self.vis_processor(image)
                image_sam = self.seg_tool.apply_image(image)  # preprocess images for sam
                resize = image_sam.shape[:2]
                image_sam = self.seg_tool.preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())

                sources = preprocess_image_text(copy.deepcopy(conversation_list),
                                                cur_token_len=self.image_token_len)

                data_dict = preprocess(sources, self.tokenizer, self.conv_type)
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                 labels=data_dict["labels"][0])

                seg_mask = torch.rand(0, *raw_size)
                boxes = torch.rand(0, 4)

                image_dict = {'image': image_clip, 'image_sam': image_sam, 'seg_mask': seg_mask, "boxes": boxes,
                              'raw_size': raw_size, 'resize': resize}

                data_dict.update(image_dict)
            except Exception as error:
                image_path = self.annotation[index]['image'] if 'image' in self.annotation[index] else str(index)
                print(f"Failed to load example {image_path}, Error: {error}. "
                      f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        return data_dict

