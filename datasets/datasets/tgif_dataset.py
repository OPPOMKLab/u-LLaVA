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
import copy
import random
from datasets.datasets.base_dataset import BaseDataset, preprocess, preprocess_video_text


class TgifDataset(BaseDataset):
    def __init__(self,
                 vis_processor,
                 tokenizer,
                 vis_root,
                 ann_root,
                 portion=1,
                 image_token_len=256,
                 data_type='video',
                 conv_type='conv_simple'
                 ):
        """
        vis_root (string): Root directory of images
        ann_root (string): Root directory of annotations
        """
        super().__init__(vis_processor=vis_processor,
                         tokenizer=tokenizer,
                         vis_root=vis_root,
                         ann_root=ann_root,
                         portion=portion,
                         data_type=data_type,
                         conv_type=conv_type
                         )

        self.resize_size = self.vis_processor.image_size
        self.num_frm = self.vis_processor.n_frm
        # temporal token (n_frm) + spatial token (num_patch)
        self.image_token_len = self.num_frm + image_token_len

    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]

                conversation_list = sample['conversations']

                if 'gif' in sample:
                    gif_path = os.path.join(self.vis_root, sample['gif'])
                    gif = self.vis_processor(gif_path)
                    # add <DEFAULT_IMAGE_PATCH_TOKEN>
                    sources = preprocess_video_text(copy.deepcopy(conversation_list),
                                                    cur_token_len=self.image_token_len)
                else:
                    gif = None
                    sources = [copy.deepcopy(conversation_list)]

                data_dict = preprocess(sources, self.tokenizer, self.conv_type)

                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                 labels=data_dict["labels"][0])

                # video exist: process by CLIP, non video: zero tensor
                if gif is not None:
                    data_dict['video'] = gif
            except:
                video_path = self.annotation[index]['gif'] if 'gif' in self.annotation[index] else str(index)
                print(f"Failed to load examples with video: {video_path}. "
                      f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        return data_dict
