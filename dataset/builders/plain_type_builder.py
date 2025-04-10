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

from utils.registry import registry
from dataset.datasets.tgif_dataset import TgifDataset
from dataset.builders.base_builder import BaseDatasetBuilder
from dataset.datasets.llava_dataset import LLaVADataset, LLaVASegDataset


class PlainBuilder(BaseDatasetBuilder):
    dataset_cls = LLaVADataset

    def build(self, tokenizer, processor_dict, conv_type='conv_simple'):

        build_info = self.config.build_info
        dataset_cls = self.dataset_cls

        image_token_len = self.config.get('image_token_len', 256)
        image_dir = build_info.get('image_dir', '')
        anno_dir = build_info.get('anno_dir', '')
        portion = float(build_info.get('portion', 1))
        data_type = self.config.get('data_type', 'image')

        vis_processor = self.fetch_processor('vis_processor', processor_dict)

        dataset = dataset_cls(
            vis_processor=vis_processor,
            tokenizer=tokenizer,
            vis_root=image_dir,
            ann_root=anno_dir,
            portion=portion,
            image_token_len=image_token_len,
            data_type=data_type,
            conv_type=conv_type,
        )

        return dataset


@registry.register_builder("llava_cc3m")
@registry.register_builder("llava_instruct")
@registry.register_builder("sqa")
class LLaVACc3mBuilder(PlainBuilder):
    dataset_cls = LLaVADataset


@registry.register_builder("llava_seg")
class LlaVASegBuilder(PlainBuilder):
    dataset_cls = LLaVASegDataset


@registry.register_builder("tgif")
class TgifBuilder(PlainBuilder):
    dataset_cls = TgifDataset


