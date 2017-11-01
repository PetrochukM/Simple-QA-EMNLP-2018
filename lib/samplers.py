import random

from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler


class SortedSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, sort_key):
        self.data_source = data_source
        self.sort_key = sort_key
        zip = [(i, self.sort_key(row)) for i, row in enumerate(self.data_source)]
        zip = sorted(zip, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data_source)


class BatchSamplerShuffle(BatchSampler):

    def __iter__(self):
        batches = list(super().__iter__())
        random.shuffle(batches)
        return iter(batches)
