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

import numpy as np
from typing import Sequence
from torch.utils.data import Subset
from torch.utils.data import Dataset
from dataset.datasets.base_dataset import BaseDataset
from torch.utils.data import ConcatDataset as TorchConcatDataset


class ConcatDataset(Dataset):

    def __init__(self, datasets: Sequence[BaseDataset]):
        self.concat_dataset = TorchConcatDataset(datasets)

    def __len__(self):
        return len(self.concat_dataset)

    def __getitem__(self, index):
        return self.concat_dataset[index]


class ConcatDatasetWithShuffle(Subset):
    def __init__(self, datasets: Sequence[BaseDataset],
                 seed=42,
                 portion=1):
        self.seed = seed
        self.portion = portion

        dataset = TorchConcatDataset(datasets)
        target_len = int(len(dataset) * portion)
        indices = list(range(len(dataset))) * int(np.ceil(portion))
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        indices = indices[:target_len]
        super().__init__(dataset, indices)
