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

from datasets.builders.base_builder import BaseDatasetBuilder
from datasets.builders.template_type_builder import RefcocoBuilder, Ade20kBuilder, \
    CocoStuffBuilder, PacoBuilder, Msra10kBuilder, ValResBuilder, ValSalientBuilder
from datasets.builders.plain_type_builder import LLaVACc3mBuilder, TgifBuilder


__all__ = [
    "LLaVACc3mBuilder",
    "TgifBuilder",
    "RefcocoBuilder",
    "Ade20kBuilder",
    "CocoStuffBuilder",
    "PacoBuilder",
    "Msra10kBuilder",
    "ValResBuilder",
    "ValSalientBuilder"
]

