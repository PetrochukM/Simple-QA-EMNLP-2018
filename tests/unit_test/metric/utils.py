import random

import torch
import torch.nn.functional as F

from torch.autograd import Variable

from seq2seq.metrics import Metric
from tests.lib.utils import MockBatch


def get_confidence(prediction, vocab_size):
    """
    Given a prediction index and vocab_size create a confidence tensor.

    Args:
        prediction (int): int that was predicted
        vocab_size (int): size of vocabulary to predict from
    Returns:
        (list of floats):
            sum(list) == 1
            the highest float is at the index `prediction`
    """
    assert prediction < vocab_size, "Prediction must smaller than the vocab."
    assert prediction >= 0, "Prediction cannot be negative."

    # Get prediction confidence
    prediction_confidence = random.randint(51, 100)
    total = 100 - prediction_confidence

    # Get the rest of the confidences
    confidences = []
    for i in range(vocab_size - 1):
        if i == vocab_size - 2:
            confidence = total
        else:
            confidence = random.randint(0, total)
        total = total - confidence
        confidences.append(confidence)

    # Add then together
    ret = []
    for i in range(vocab_size):
        if i == prediction:
            ret.append(round(prediction_confidence / 100.0, 2))
        else:
            ret.append(round(confidences.pop() / 100.0, 2))

    assert round(sum(ret), 2) == 1

    return ret


def get_random_batch(output_seq_len, input_seq_len, batch_size, output_field, input_field):
    """
    Motivation: Get a random batch to compute the loss on.

    Args:
        output_seq_len (int): length of the sequence
        batch_size (int): size of the batch
        output_field (torchtext.data.Field): field used to process the output
    Returns:
        outputs (torch.FloatTensor [output_seq_len, batch_size, dictionary_size]): random outputs of a
            batch.
        targets (torch.LongTensor [seq_output_seq_lenlen, batch_size]): random expected output of a batch.
    """
    outputs = torch.stack([
        F.softmax(Variable(torch.randn(output_seq_len, len(output_field.vocab))))
        for _ in range(batch_size)
    ])
    targets = torch.stack([
        Variable(
            torch.LongTensor(
                [random.randint(0, len(output_field.vocab) - 1) for _ in range(output_seq_len)]))
        for _ in range(batch_size)
    ])
    inputs = torch.stack([
        Variable(
            torch.LongTensor(
                [random.randint(0, len(input_field.vocab) - 1) for _ in range(input_seq_len)]))
        for _ in range(batch_size)
    ])
    outputs = outputs.transpose(0, 1).contiguous()
    targets = targets.transpose(0, 1).contiguous()
    inputs = inputs.transpose(0, 1).contiguous()
    assert outputs.size() == (output_seq_len, batch_size, len(output_field.vocab))
    assert targets.size() == (output_seq_len, batch_size)
    assert inputs.size() == (input_seq_len, batch_size)
    return outputs, MockBatch(output=[targets, []], input_=[inputs, []])


def get_batch(predictions, targets, sources=None, vocab_size=3):
    """
    Given predictions and targets, create a batch that matches those.

    Args:
        predictions (list of lists): Batch size number of lists of sequences.
        targets (list of lists): Batch size number of lists of sequences.
        sources (list of lists, optional): Batch size number of lists of sequences.
    Returns:
        outputs (torch.Tensor seq_len x batch_size x dictionary_size): outputs of a batch.
        batch (MockBatch)
    """
    if sources is None:
        sources = targets

    assert len(predictions) == len(targets), "Targets batchsize should be the same as predictions"
    assert len(targets) == len(sources), "Sources batchsize should be the same as predictions"
    outputs = []
    for sequence in predictions:
        output = [get_confidence(prediction, vocab_size) for prediction in sequence]
        outputs.append(output)

    batch_size = len(predictions)
    output_seq_len = len(predictions[0])
    input_seq_len = len(sources[0])
    for i in range(batch_size):
        assert len(targets[i]) == output_seq_len, "Sizes need to be consistent"
        assert len(predictions[i]) == output_seq_len, "Sizes need to be consistent"
    sources = Variable(torch.LongTensor(sources).transpose(0, 1).contiguous())
    targets = Variable(torch.LongTensor(targets).transpose(0, 1).contiguous())
    outputs = Variable(torch.FloatTensor(outputs).transpose(0, 1).contiguous())

    assert sources.size() == (input_seq_len, batch_size)
    assert targets.size() == (output_seq_len, batch_size)
    assert outputs.size() == (output_seq_len, batch_size, vocab_size)

    return outputs, MockBatch(output=[targets, []], input_=[sources, []])


# Used to test get_metrics adding external metrics
class MockMetric(Metric):

    def __init__(self):
        pass
