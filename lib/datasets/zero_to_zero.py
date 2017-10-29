import torchtext

from seq2seq.datasets.seq_input_output_dataset import SeqInputOutputDataset


class ZeroToZero(SeqInputOutputDataset):

    def __init__(self, n_rows, input_field, output_field):
        fields = [('input', input_field), ('output', output_field)]
        examples = []
        for i in range(n_rows):
            example = torchtext.data.Example.fromlist([str(0), str(0)], fields)
            examples.append(example)
        super().__init__(examples, fields)

    @classmethod
    def splits(cls, input_field, output_field, train=256, dev=64, test=64):
        """
        Missing function definition can be found in `SeqInputOutputDataset`.
        """

        def make(n_rows):
            if not n_rows:
                return None
            return cls(n_rows, input_field, output_field)

        return tuple(d for d in (make(train), make(dev), make(test)) if d is not None)
