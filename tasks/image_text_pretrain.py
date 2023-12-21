"""
Partially Adapted form: https://github.com/DAMO-NLP-SG/Video-LLaMA/blob/main/video_llama/tasks/image_text_pretrain.py
"""

from utils.registry import registry
from tasks.base_task import BaseTask
from utils.tools import datetime_print
from datasets.datasets.concat_dataset import ConcatDataset, ConcatDatasetWithShuffle


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self, cfg):
        super().__init__(cfg)

    @staticmethod
    def build_datasets(datasets_config, tokenizer, processor_dict, conv_type='conv_simple'):
        """

        :param datasets_config:
        :param tokenizer:
        :param processor_dict: {'clip_image': CLIPImageProcessor()}
        :param conv_type:
        :return:

        """

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        if len(datasets_config) == 1:
            name = list(datasets_config.keys())[0]
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)

            # {"train": dataset, "test": dataset}
            dataset = builder.build(tokenizer, processor_dict, conv_type)

        else:
            shuffle = True
            portion = 1
            dataset_list = []

            for idx, name in enumerate(datasets_config):
                datetime_print('BUILDING DATASET {0}: {1}'.format(idx+1, name))
                dataset_config = datasets_config[name]

                builder = registry.get_builder_class(name)(dataset_config)
                current_dataset = builder.build(tokenizer, processor_dict, conv_type)

                dataset_list.append(current_dataset)

            if shuffle:
                dataset = ConcatDatasetWithShuffle(dataset_list, portion=portion)
            else:
                dataset = ConcatDataset(dataset_list)

        return dataset
