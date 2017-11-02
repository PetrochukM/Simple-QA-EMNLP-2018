from functools import partial

import argparse
import logging
import os
import time

from torch.nn.modules.loss import NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

import torch
import pandas as pd

from lib.checkpoint import Checkpoint
from lib.configurable import add_config
from lib.configurable import configurable
from lib.datasets import reverse
from lib.nn import DecoderRNN
from lib.nn import EncoderRNN
from lib.nn import SeqToSeq
from lib.metrics import Accuracy
from lib.metrics import RandomSample
from lib.optim import Optimizer
from lib.samplers import BucketBatchSampler
from lib.text_encoders import PADDING_INDEX
from lib.text_encoders import COPY_INDEX
from lib.text_encoders import WordEncoder
from lib.utils import collate_fn
from lib.utils import device_default
from lib.utils import init_logging
from lib.utils import add_logger_file_handler
from lib.utils import get_total_parameters
from lib.utils import seed

init_logging()
logger = logging.getLogger(__name__)  # Root logger

Adam.__init__ = configurable(Adam.__init__)
StepLR.__init__ = configurable(StepLR.__init__)

pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 80)

# NOTE: The goal of this file is just to setup the training for simple_questions_predicate.

# TODO: In order to not copy code this should be implemented as a branch

BASE_RNN_HYPERPARAMETERS = {
    'embedding_size': 128,
    'rnn_size': 64,
    'n_layers': 1,
    'rnn_cell': 'lstm',
    'embedding_dropout': 0.0,
    'rnn_variational_dropout': 0.0,
    'rnn_dropout': 0.4,
}

DEFAULT_HYPERPARAMETERS = {
    'lib': {
        'nn': {
            'decoder_rnn.DecoderRNN.__init__': {
                'use_attention': True,
                'scheduled_sampling': False,
            },
            'encoder_rnn.EncoderRNN.__init__': {
                'bidirectional': True,
            },
            'attention.Attention.__init__.attention_type': 'general',
        },
        'optim.optim.Optimizer.__init__': {
            'max_grad_norm': 1.0,
        }
    },
    'torch.optim.Adam.__init__': {
        'lr': 0.001,
        'weight_decay': 0,
    }
}

DEFAULT_HYPERPARAMETERS['lib']['nn']['decoder_rnn.DecoderRNN.__init__'].update(
    BASE_RNN_HYPERPARAMETERS)
DEFAULT_HYPERPARAMETERS['lib']['nn']['encoder_rnn.EncoderRNN.__init__'].update(
    BASE_RNN_HYPERPARAMETERS)

DEFAULT_SAVE_DIRECTORY_NAME = __file__.split('/')[-1].replace('.py', '')
DEFAULT_SAVE_DIRECTORY_NAME += time.strftime('_%mm_%dd_%Hh_%Mm_%Ss', time.localtime())
DEFAULT_SAVE_DIRECTORY = os.path.join('save/', DEFAULT_SAVE_DIRECTORY_NAME)


def main(
        dataset=reverse,
        checkpoint_path=None,
        save_directory=DEFAULT_SAVE_DIRECTORY,
        device=None,
        random_seed=123,  # Reproducibility
        epochs=4,
        train_batch_size=16,
        dev_batch_size=128):
    # Save a copy of all logger logs to `save_directory`/train.log
    filename = os.path.join(save_directory, 'train.log')
    add_logger_file_handler(filename)

    add_config(DEFAULT_HYPERPARAMETERS)

    # Setup Device
    device = device_default(device)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    logger.info('Device: %s', device)

    # Random Seed for reproducibility
    seed(random_seed)

    # Load Checkpoint
    checkpoint = Checkpoint.load(
        checkpoint_path=checkpoint_path, save_directory=save_directory, device=device)

    # Init Dataset
    train_dataset, dev_dataset = dataset(train=True, dev=True, test=False)
    # NOTE: `source` and `target` are typically associated with sequences
    assert 'source' in train_dataset and 'source' in dev_dataset
    assert 'target' in train_dataset and 'target' in dev_dataset

    logger.info('Num Training Data: %d', len(train_dataset))
    logger.info('Num Development Data: %d', len(dev_dataset))

    # Init Encoders and encode dataset
    if checkpoint:
        source_encoder, target_encoder = checkpoint.input_encoder, checkpoint.output_encoder
    else:
        # For the experiment, we require the dictionaries are the same
        source_encoder = WordEncoder(train_dataset['source'] + train_dataset['target'])
        target_encoder = source_encoder

    for dataset in [train_dataset, dev_dataset]:
        for row in dataset:
            row['source'] = source_encoder.encode(row['source'])
            row['target'] = target_encoder.encode(row['target'])

    # Init Model
    if checkpoint:
        model = checkpoint.model
    else:
        model = SeqToSeq(
            EncoderRNN(vocab_size=source_encoder.vocab_size, embeddings=source_encoder.embeddings),
            DecoderRNN(vocab_size=target_encoder.vocab_size, embeddings=target_encoder.embeddings))
        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)

    logger.info('Model:\n%s', model)
    logger.info('Total Parameters: %d', get_total_parameters(model))

    if torch.cuda.is_available():
        model.cuda(device_id=device)

    # Init Loss
    criterion = NLLLoss(ignore_index=PADDING_INDEX)
    if torch.cuda.is_available():
        criterion.cuda()

    # Init Optimizer
    # https://github.com/pytorch/pytorch/issues/679
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Optimizer(Adam(params=params))
    optimizer.set_scheduler(StepLR)

    # Init Observers
    dev_observers = [
        Accuracy(ignore_index=PADDING_INDEX), RandomSample(
            source_encoder, target_encoder, n_samples=5, ignore_index=PADDING_INDEX)
    ]

    # collate function merges rows into tensor batches
    collate_fn_partial = partial(
        collate_fn, input_key='source', output_key='target', sort_key='source')

    def prepare_batch(batch):
        # Prepare batch for model
        source, source_lengths = batch['source']
        target, target_lengths = batch['target']
        if torch.cuda.is_available():
            source, source_lengths = source.cuda(async=True), source_lengths.cuda(async=True)
            target, target_lengths = target.cuda(async=True), target_lengths.cuda(async=True)
        source = Variable(source)
        target = Variable(target)
        return source, source_lengths, target, target_lengths

    # Train!
    for i in range(epochs):
        logger.info('Starting epoch %d', i)
        model.train(mode=True)
        batch_sampler = BucketBatchSampler(train_dataset, lambda r: r['source'].size()[0],
                                           train_batch_size)
        for batch in tqdm(
                DataLoader(
                    train_dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=collate_fn_partial,
                    pin_memory=torch.cuda.is_available(),
                    num_workers=0)):
            source, source_lengths, target, target_lengths = prepare_batch(batch)

            optimizer.zero_grad()
            output, _, attention = model(source, source_lengths, target, target_lengths)

            # Given the max attention weight selects the correct `source_value` change the 
            # `target` to COPY_INDEX; otherwise, keep it consistent. 
            attention = attention.max(2)[1]  # [batch_size, output_length]
            for target_index in range(target.size()[0]):
                for batch_index in range(target.size()[1]):
                    source_index = attention[target_index, batch_index].data[0]
                    source_value = source[source_index, batch_index]
                    if target[target_index, batch_index].data[0] == source_value.data[0]:
                        target[target_index, batch_index] = COPY_INDEX

            # Compute loss
            # NOTE: flattening the tensors allows for the computation to be done once per batch
            output_flat = output.view(-1, output.size(2))
            target_flat = target.view(-1)
            loss = criterion(output_flat, target_flat)

            # Backward propagation
            loss.backward()
            optimizer.step()

        Checkpoint.save(
            save_directory=save_directory,
            model=model,
            optimizer=optimizer,
            input_text_encoder=source_encoder,
            output_text_encoder=target_encoder,
            device=device)

        model.train(mode=False)
        # Train then evaluate
        copy_index_tokens = 0
        total_tokens = 0
        for batch in DataLoader(
                dev_dataset,
                batch_size=dev_batch_size,
                collate_fn=collate_fn_partial,
                pin_memory=torch.cuda.is_available(),
                num_workers=0):
            source, source_lengths, target, target_lengths = prepare_batch(batch)
            output, _, attention = model(source, source_lengths, target, target_lengths)

            # Given a COPY_INDEX was predicted replace it with the copy value.
            output = output.data
            attention = attention.max(2)[1]  # [batch_size, output_length]
            ignore_index = PADDING_INDEX
            for output_index in range(output.size()[0]):
                for batch_index in range(output.size()[1]):
                    if output[output_index, batch_index].max(0)[1][0] == COPY_INDEX:
                        source_index = attention[output_index, batch_index].data[0]
                        source_value = source[source_index, batch_index].data[0]
                        # Change the distribution to predict the copy value
                        output[output_index, batch_index, source_value] = output[
                            output_index, batch_index, COPY_INDEX]
                        output[output_index, batch_index, COPY_INDEX] = -1000
                        if source_value != ignore_index:
                            copy_index_tokens += 1
                            total_tokens += 1
                    elif output[output_index, batch_index].max(0)[1][0] != ignore_index:
                        total_tokens += 1

            for observer in dev_observers:
                observer.update({
                    'source_batch': source,
                    'target_batch': target,
                    'output_batch': Variable(output)
                })

        logger.info('Frequency of copy tokens: %.03f [%d of %d]', copy_index_tokens / total_tokens,
                    copy_index_tokens, total_tokens)
        [observer.dump(save_directory).reset() for observer in dev_observers]


if __name__ == '__main__':
    """
    Simple Questions Predicate CLI.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint_path', type=str, default=None, help="Path to checkpoint to resume training.")
    args = parser.parse_args()
    save_directory = os.path.dirname(
        args.checkpoint_path) if args.checkpoint_path else DEFAULT_SAVE_DIRECTORY
    os.makedirs(save_directory)
    main(checkpoint_path=args.checkpoint_path, save_directory=save_directory)
