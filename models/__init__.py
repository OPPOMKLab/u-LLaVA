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

from models.ullava import UllavaConfig, UllavaForCausalLM
from models.ullava_core import UllavaCoreConfig, UllavaCoreForCausalLM
from models.grounding_module import load_groundingdino_model, GroundingModule
from models.tools import KeywordsStoppingCriteria, smart_resize_token_embedding, \
    smart_special_token_and_embedding_resize, multi_modal_resize_token_embedding

DEFAULT_IMG_TOKEN = '<image>'

DEFAULT_IMG_PATCH_TOKEN = "<image_patch>"
DEFAULT_IMG_START_TOKEN = "<img_beg>"
DEFAULT_IMG_END_TOKEN = "</img_end>"

DEFAULT_VID_PATCH_TOKEN = "<video_patch>"
DEFAULT_VID_START_TOKEN = "<vid_beg>"
DEFAULT_VID_END_TOKEN = "</vid_end>"

DEFAULT_SEG_TOKEN = '[SEG]'
DEFAULT_TAG_START = '[tag]'
DEFAULT_TAG_END = '[/tag]'

DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_UNK_TOKEN = '<unk>'
DEFAULT_PAD_TOKEN = '[PAD]'
IGNORE_INDEX = -100


__all__ = [
    "UllavaConfig",
    "UllavaForCausalLM",
    "UllavaCoreConfig",
    "UllavaCoreForCausalLM",
    "GroundingModule",
    "load_groundingdino_model",
    "KeywordsStoppingCriteria",
    "smart_resize_token_embedding",
    "multi_modal_resize_token_embedding",
    "smart_special_token_and_embedding_resize",
    "DEFAULT_IMG_TOKEN",
    "DEFAULT_SEG_TOKEN",
    "DEFAULT_IMG_PATCH_TOKEN",
    "DEFAULT_IMG_START_TOKEN",
    "DEFAULT_IMG_END_TOKEN",
    "DEFAULT_VID_PATCH_TOKEN",
    "DEFAULT_VID_START_TOKEN",
    "DEFAULT_VID_END_TOKEN",
    "DEFAULT_BOS_TOKEN",
    "DEFAULT_EOS_TOKEN",
    "DEFAULT_UNK_TOKEN",
    "DEFAULT_PAD_TOKEN",
    "IGNORE_INDEX",
    "DEFAULT_TAG_START",
    "DEFAULT_TAG_END"
]



