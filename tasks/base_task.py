"""
Partially Adapted form: https://github.com/DAMO-NLP-SG/Video-LLaMA/blob/main/video_llama/tasks/base_task.py
"""

from utils.registry import registry
from utils.tools import datetime_print


class BaseTask:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def build_model(model_cfg):
        model_cls = registry.get_model_class(model_cfg.arch)
        return model_cls.from_config(model_cfg)

    def build_collator(self, pad_token_id):
        """
        :param pad_token_id: tokenizer.pad_token_id
        :return: data collator
        """
        collator_type = self.cfg.get('collator_type', 'base_collator')
        data_collator = registry.get_collator_class(collator_type)(pad_token_id)
        return data_collator

    @staticmethod
    def build_processors(processors_cfg):
        """
        :param processors_cfg:
            processor:
                clip_image:
                    path:
                    image_size: 224
                video_train:
                    n_frm: 8
                    image_size: 224
                gif_train:
                    n_frm: 8
                    image_size: 224
                plain_box:
                    precision: 3
        :return:
        """
        processors = dict()
        for idx, name in enumerate(processors_cfg):
            datetime_print('BUILDING PROCESSOR {0}: {1}'.format(idx + 1, name))
            processor_cfg = processors_cfg[name]
            processor = registry.get_processor_class(name).from_config(processor_cfg)
            processors[name] = processor

        return processors

    @staticmethod
    def build_datasets(datasets_config, tokenizer, processor_dict, conv_type='conv_simple'):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.

        :param datasets_config:
                    dataset_1
                        image_dir
                    dataset_2
                        image_dir
        :param tokenizer:
        :param processor_dict: {'clip_image': CLIPImageProcessor()}
        :param conv_type: 'conv_simple'
        Returns:
            Dictionary of torch.utils.data.Dataset objects by split.

            datasets: {
                'llava_instruct': {'train': dataset, 'test': dataset},
                'para_instruct': {'train': dataset, 'test': dataset}
            }
        """
        datasets = dict()

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build(tokenizer, processor_dict, conv_type)

            datasets[name] = dataset

        return datasets

