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
from dataset.builders.base_builder import BaseDatasetBuilder
from dataset.datasets.res_dataset import ResDataset, ValResDataset
from dataset.datasets.salient_seg_dataset import SalientSegDataset, ValSalientSegDataset
from dataset.datasets.sem_seg_dataset import SemanticSegDataset, CocoStuffDataset, PacoDataset


class TemplateBuilder(BaseDatasetBuilder):
    dataset_cls = None

    def build(self, tokenizer, processor_dict, conv_type='conv_simple'):
        build_info = self.config.build_info
        dataset_cls = self.dataset_cls

        image_token_len = self.config.get('image_token_len', 256)
        image_dir = build_info.get('image_dir', '')
        anno_dir = build_info.get('anno_dir', '')
        portion = float(build_info.get('portion', 1))
        template_root = build_info.get('template_root', '')
        data_type = self.config.get('data_type', 'image')

        vis_processor = self.fetch_processor('vis_processor', processor_dict)

        dataset = dataset_cls(
            vis_processor=vis_processor,
            tokenizer=tokenizer,
            vis_root=image_dir,
            ann_root=anno_dir,
            template_root=template_root,
            portion=portion,
            image_token_len=image_token_len,
            data_type=data_type,
            conv_type=conv_type
        )

        return dataset


@registry.register_builder('refcoco')
@registry.register_builder('refcoco+')
@registry.register_builder('refcocog')
@registry.register_builder('refclef')
class RefcocoBuilder(TemplateBuilder):
    dataset_cls = ResDataset


@registry.register_builder('ade20k')
class Ade20kBuilder(TemplateBuilder):
    dataset_cls = SemanticSegDataset


@registry.register_builder('cocostuff')
class CocoStuffBuilder(TemplateBuilder):
    dataset_cls = CocoStuffDataset


@registry.register_builder('paco_lvis')
@registry.register_builder('pascal_part')
class PacoBuilder(TemplateBuilder):
    dataset_cls = PacoDataset


@registry.register_builder('msra_10k')
@registry.register_builder('msra_b')
class Msra10kBuilder(TemplateBuilder):
    dataset_cls = SalientSegDataset


@registry.register_builder('refcoco_val')
@registry.register_builder('refcoco_testA')
@registry.register_builder('refcoco_testB')
@registry.register_builder('refcoco+_val')
@registry.register_builder('refcoco+_testA')
@registry.register_builder('refcoco+_testB')
@registry.register_builder('refcocog_val')
@registry.register_builder('refcocog_test')
class ValResBuilder(TemplateBuilder):
    dataset_cls = ValResDataset


@registry.register_builder('dut_omron')
@registry.register_builder('duts_te')
@registry.register_builder('ecssd')
class ValSalientBuilder(TemplateBuilder):
    dataset_cls = ValSalientSegDataset
