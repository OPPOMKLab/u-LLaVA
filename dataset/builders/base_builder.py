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
import warnings
from omegaconf import OmegaConf


class BaseDatasetBuilder:
    """
    builders of preprocessor and dataset
    related to: processors, datasets
    """
    dataset_cls = None

    def __init__(self, cfg=None):
        """
        :param cfg: full config, including models, datasets, etc.
        """
        super().__init__()

        if isinstance(cfg, str):
            # load from path
            self.config = self.load_dataset_config(cfg)
        else:
            # when called from task.build_dataset()
            self.config = cfg

    @staticmethod
    def load_dataset_config(cfg_path):
        cfg = OmegaConf.load(cfg_path).dataset
        cfg = cfg[list(cfg.keys())[0]]

        return cfg

    def fetch_processor(self, processor_type='vis_processor', processor_dict=None):
        """

        :param processor_type: 'vis_processor' or 'box_processor'
        :param processor_dict: {'clip_image': CLIPImageProcessor()}
        :return:
        """
        name = self.config.get(processor_type, None)

        return processor_dict[name] if name is not None else None

    def build(self, tokenizer, processors_dict, conv_type='conv_simple'):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """

        build_info = self.config.build_info
        ann_root = build_info.anno_dir
        vis_root = build_info.image_dir

        # processors
        vis_processor = self.fetch_processor('vis_processor', processors_dict)

        if not os.path.exists(vis_root) or not os.path.exists(ann_root):
            warnings.warn("storage path {0} or {1} does not exist.".format(vis_root, ann_root))

        # create dataset
        dataset_cls = self.dataset_cls

        dataset = dataset_cls(
            vis_processor=vis_processor,
            tokenizer=tokenizer,
            vis_root=vis_root,
            ann_root=ann_root,
            conv_type=conv_type
        )

        return dataset
