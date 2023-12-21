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

from PIL import Image
from utils.registry import registry
from transformers import CLIPImageProcessor
from datasets.processors.base_processor import BaseProcessor


def expand2square(pil_img, background_color=(255, 255, 255)):
    """
    Fill background for PIL image to aspect ratio 1
    :param pil_img:
    :param background_color:
    :return:
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


@registry.register_processor('clip_image')
class CLIPProcessor(BaseProcessor):
    def __init__(self, checkpoint_path, aspect_ratio=None):
        """
        :param checkpoint_path:
        :param aspect_ratio: 长宽比 ['pad', 'keep', None]
        """
        super(CLIPProcessor, self).__init__()
        self.image_processor = CLIPImageProcessor.from_pretrained(checkpoint_path)
        self.aspect_ratio = aspect_ratio

    def expand_image(self, pil_img, background_color=(255, 255, 255)):
        """
        :param pil_img: PIL.Image, RGB
        :param background_color:
        :return:
        """
        if self.aspect_ratio is None:
            return pil_img
        elif self.aspect_ratio == 'pad':
            return expand2square(pil_img, background_color)
        else:
            raise NotImplementedError

    def __call__(self, item):
        """
        :param item: PIL.Image, RGB
        :return:
        """

        image = self.image_processor.preprocess(item, return_tensors='pt')['pixel_values'][0]

        return image

    @classmethod
    def from_config(cls, cfg=None):
        checkpoint_path = cfg.get("path", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        return cls(checkpoint_path, aspect_ratio)
