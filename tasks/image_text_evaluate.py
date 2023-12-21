"""
Partially Adapted form: https://github.com/DAMO-NLP-SG/Video-LLaMA/blob/main/video_llama/tasks/image_text_pretrain.py
"""

from utils.registry import registry
from tasks.base_task import BaseTask
from utils.tools import datetime_print


@registry.register_task("image_text_evaluate")
class ImageTextEvaluateTask(BaseTask):
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

        dataset_dict = {}

        for idx, name in enumerate(datasets_config):
            datetime_print('BUILDING DATASET {0}: {1}'.format(idx+1, name))
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            current_dataset = builder.build(tokenizer, processor_dict, conv_type)

            dataset_dict[name] = current_dataset

        return dataset_dict
