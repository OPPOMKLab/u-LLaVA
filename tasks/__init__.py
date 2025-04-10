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
from tasks.base_task import BaseTask
from tasks.image_text_pretrain import ImageTextPretrainTask
from tasks.image_text_evaluate import ImageTextEvaluateTask
# do not delete
from dataset.builders import *
from dataset.collators import *
from dataset.processors import *


def setup_task(cfg):
    task = registry.get_task_class(cfg.type)(cfg)
    assert task is not None, "Task {} not properly registered.".format(cfg.type)

    return task


__all__ = [
    "BaseTask",
    "ImageTextPretrainTask",
    "ImageTextEvaluateTask",
    "setup_task"
]
