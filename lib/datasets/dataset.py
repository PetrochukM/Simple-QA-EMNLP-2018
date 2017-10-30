from torch.utils import data


# TODO: Verify this is a 2D dataset with no holes?
# NOTE: It's not valuable to verify that it has no holes cause that'll come up in training easily
class Dataset(data.Dataset):

    def __init__(self, rows):
        self.columns = set(rows[0].keys())
        for row in rows:
            assert isinstance(row, object)
            assert self.columns == set(row.keys())
        self.rows = rows

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.rows[key]
        elif isinstance(key, str):
            return [row[key] for row in self.rows]
        else:
            raise AttributeError

    def __len__(self):
        return len(self.rows)

    def __contains__(self, key):
        return key in self.columns


# Missing: Using PyTorch dataloaders
class SeqToClassDataset(object):
    pass

    # DEFINES: The iterator or table? Defines the columns of the batch?
    # MISSING: If a model requires a classification dataset, the dataset has to define a label
    # and text in it's batches during iteration. 

    # Batch can be defined with multiple outputs as long as there is a text and label for the model
    # to refer too? What about more complicated achitectures; then there more fields per row.
    # FOR example, the french and english can have a source and destination


class SeqToSeqDataset(object):
    pass
