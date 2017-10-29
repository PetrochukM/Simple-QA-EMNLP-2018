import torchtext
import random

from seq2seq.datasets.seq_input_output_dataset import SeqInputOutputDataset


class Reverse(SeqInputOutputDataset):
    generate_max_len = 10  # Max length of space delimited numbers to reverse

    def __init__(self, n_rows, input_field, output_field):
        fields = [('input', input_field), ('output', output_field)]
        examples = []
        for i in range(n_rows):
            length = random.randint(1, Reverse.generate_max_len)
            seq = []
            for _ in range(length):
                seq.append(str(random.randint(0, 9)))
            input_ = ' '.join(seq)
            output = ' '.join(reversed(seq))
            example = torchtext.data.Example.fromlist([input_, output], fields)
            examples.append(example)
        super().__init__(examples, fields)

    @classmethod
    def splits(cls, input_field, output_field, train=10000, dev=1000, test=1000):
        """
        Missing function definition can be found in `SeqInputOutputDataset`.
        """

        def make(n_rows):
            if not n_rows:
                return None
            return cls(n_rows, input_field, output_field)

        return tuple(d for d in (make(train), make(dev), make(test)) if d is not None)
