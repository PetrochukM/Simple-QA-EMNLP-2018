from torch.utils import data


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
