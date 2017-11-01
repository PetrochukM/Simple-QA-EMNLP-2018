import random

from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler


class SortedSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
        sort_key (callable): callable that returns from one row of the data_source a sortable
            value
    """

    def __init__(self, data_source, sort_key, sort_noise=0.1):
        self.data_source = data_source
        self.sort_key = sort_key
        zip = [(i, self.sort_key(row)) for i, row in enumerate(self.data_source)]
        zip = sorted(zip, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data_source)


class NoiseSortedSampler(SortedSampler):
    """Samples elements sequentially, always in the same order.

    Reference and inspiration:
    https://github.com/allenai/allennlp/blob/e125a490b71b21e914af01e70e9b00b165d64dcd/allennlp/data/iterators/bucket_iterator.py

    Arguments:
        data_source (Dataset): dataset to sample from
        sort_key (callable -> int): callable that returns from one row of the data_source a int
    """

    def __init__(self, data_source, sort_key, sort_key_noise=0.1):
        self.data_source = data_source
        self.sort_key = sort_key
        zip = []
        for i, row in enumerate(self.data_source):
            value = self.sort_key(row)
            noise_value = value * sort_key_noise
            noise = random.uniform(-noise_value, noise_value)
            value = noise + value
            zip.append(tuple([i, value]))
        zip = sorted(zip, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip]


class RandomBatchSampler(BatchSampler):

    def __iter__(self):
        batches = list(super().__iter__())
        random.shuffle(batches)
        return iter(batches)


class BucketBatchSampler(BatchSampler):
    """
    Reference:
    https://github.com/allenai/allennlp/blob/e125a490b71b21e914af01e70e9b00b165d64dcd/allennlp/data/iterators/bucket_iterator.py
    https://github.com/pytorch/text/tree/master/torchtext/data/iterators/#BucketIterator 

    `BucketIterator` pools together examples with a similar size length to reduce the padding
    required for each batch. `BucketIterator` typically also includes the ability to add noise to
    the pooling.

    The functionality has been replicated as a `Sampler` to be used with a
    `torch.data.utils.DataLoader`.
    """

    def __init__(self,
                 data_source,
                 sort_key,
                 batch_size,
                 drop_last=False,
                 sort_key_noise=0.1,
                 last_batch_first=True,
                 shuffle=True):
        self.last_batch_first = last_batch_first
        self.shuffle = shuffle
        super().__init__(
            NoiseSortedSampler(data_source, sort_key, sort_key_noise), batch_size, drop_last)

    def __iter__(self):
        batches = list(super().__iter__())
        if self.last_batch_first:
            last_batch = batches.pop()
        if self.shuffle:
            random.shuffle(batches)
        batches.insert(0, last_batch)
        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
