import torch
from torch.utils.data import Dataset

import numpy as np

# from . import utils


class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class DatasetCache(DatasetWrapper):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        else:
            self.cache[index] = self.dataset[index]
            return self.cache[index]


class DatasetSubset(DatasetWrapper):
    def __init__(self, dataset, start=0, stop=None):
        super().__init__(dataset)
        self.start = start
        self.stop = stop
        if stop is not None:
            self.stop = min(self.stop, len(self.dataset))

    def __getitem__(self, index):
        return self.dataset[index + self.start]

    def __len__(self):
        stop = self.stop or len(self.dataset)
        return stop - self.start


class DatasetWhiten(DatasetWrapper):
    def __init__(self, dataset):
        super().__init__(dataset)

        all_unwhite_x = torch.stack([self.dataset[i][0]
                                     for i in range(len(self.dataset))])
        self.mean, self.std = all_unwhite_x.mean(), all_unwhite_x.std()

    def __getitem__(self, index):
        return (self.dataset[index][0] - self.mean) / self.std, self.dataset[index][1]


class DatasetUnion(DatasetWrapper):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cum_len = np.cumsum([len(d) for d in self.datasets])

    def _find_dataset(self, index):
        match = 0
        while index >= self.cum_len[match]:
            match += 1
        return match

    def __getitem__(self, index):
        dataset_index = self._find_dataset(index)
        offset = self.cum_len[dataset_index]
        return self.datasets[dataset_index][index - offset]

    def __len__(self):
        return self.cum_len[-1]


class DatasetShuffle(DatasetWrapper):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.mapping = np.arange(len(self.dataset))
        np.random.shuffle(self.mapping)

    def __getitem__(self, index):
        return self.dataset[self.mapping[index]]


class DatasetTransform(DatasetWrapper):
    def __init__(self, dataset, transform=None, target_transform=None):
        super().__init__(dataset)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x, y = super().__getitem__(index)

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y


# class DatasetTransformCache(DatasetWrapper):
#     def __init__(self, dataset, batch_transforms, batch_size):
#         super().__init__(dataset)
#         loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
#         data_x, data_y = utils.transform_stack_data(
#             loader, batch_transforms, len(dataset))
#         self.data_x = data_x
#         self.data_y = data_y
# 
#     def __getitem__(self, index):
#         return self.data_x[index], self.data_y[index]
