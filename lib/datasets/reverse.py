import random

from lib.datasets.dataset import Dataset


def reverse(train=False,
            dev=False,
            test=False,
            train_rows=10000,
            dev_rows=1000,
            test_rows=1000,
            seq_max_length=10):
    ret = []
    for is_requested, n_rows in [(train, train_rows), (dev, dev_rows), (test, test_rows)]:
        if not is_requested:
            continue
        rows = []
        for i in range(n_rows):
            length = random.randint(1, seq_max_length)
            seq = []
            for _ in range(length):
                seq.append(str(random.randint(0, 9)))
            input_ = ' '.join(seq)
            output = ' '.join(reversed(seq))
            rows.append({'source': input_, 'target': output})
        ret.append(Dataset(rows))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
