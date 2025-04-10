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
from models import IGNORE_INDEX
from utils.registry import registry


@registry.register_collator('base_collator')
class BaseCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
        self.ignore_index = IGNORE_INDEX

    def process_text(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=self.ignore_index)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.pad_token_id),
        )
        return batch

    def __call__(self, instances):
        batch = self.process_text(instances)

        # gather all images in the batch
        images = [instance['image'] for instance in instances if 'image' in instance]
        # if one sample has no image, it will be selected out by <img_beg> in the model
        batch['images'] = torch.stack(images) if images else None

        return batch


@registry.register_collator('image_collator')
class ImageCollator(BaseCollator):
    pass


@registry.register_collator('video_collator')
class VideoCollator(BaseCollator):
    def __call__(self, instances):
        batch = self.process_text(instances)
        # gather all images in the batch
        videos = [instance['video'] for instance in instances if 'video' in instance]
        # if one sample has no video, it will be selected out by <vid_beg> in the model
        batch['videos'] = torch.stack(videos) if videos else None

        return batch


@registry.register_collator('image_video_collator')
class ImageVideoCollator(BaseCollator):
    def __call__(self, instances):
        batch = self.process_text(instances)

        # gather image
        images = [instance['image'] for instance in instances if 'image' in instance]
        batch['images'] = torch.stack(images) if images else None

        # gather video
        videos = [instance['video'] for instance in instances if 'video' in instance]
        batch['videos'] = torch.stack(videos) if videos else None

        return batch


@registry.register_collator('segmentation_collator')
class SegmentationCollator(BaseCollator):
    def __call__(self, instances):
        batch = self.process_text(instances)

        images = [instance['image'] for instance in instances]
        batch['images'] = torch.stack(images)

        images_sam = [instance['image_sam'] for instance in instances]
        batch['images_sam'] = torch.stack(images_sam)

        batch['mask_list'] = [instance['seg_mask'] for instance in instances]
        batch['size_list'] = [instance['raw_size'] for instance in instances]
        batch['resize_list'] = [instance['resize'] for instance in instances]

        return batch


@registry.register_collator('grounding_collator')
class GroundingCollator(BaseCollator):
    def __call__(self, instances):
        batch = self.process_text(instances)

        images = [instance['image'] for instance in instances]
        batch['images'] = torch.stack(images)

        images_sam = [instance['image_sam'] for instance in instances]
        batch['images_sam'] = torch.stack(images_sam)

        batch['mask_list'] = [instance['seg_mask'] for instance in instances]
        batch['size_list'] = [instance['raw_size'] for instance in instances]
        batch['resize_list'] = [instance['resize'] for instance in instances]
        batch['bbox_list'] = [instance['boxes'] for instance in instances]

        return batch

