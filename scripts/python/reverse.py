from functools import partial

import argparse
import logging
import os
import random
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
from lib.nn import Seq2seq
from lib.observers import Accuracy
from lib.observers import RandomSample
from lib.optim import Optimizer
from lib.samplers import SortedSampler
from lib.samplers import BatchSamplerShuffle
from lib.text_encoders import PADDING_INDEX
from lib.text_encoders import WordEncoder
from lib.utils import collate_fn
from lib.utils import device_default
from lib.utils import init_logging

init_logging()
logger = logging.getLogger()  # Root logger

Adam.__init__ = configurable(Adam.__init__)
StepLR.__init__ = configurable(StepLR.__init__)

# NOTE: The goal of this file is just to setup the training for simple_questions_predicate.

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
                'scheduled_sampling': True
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

DEFAULT_SAVE_DIRECTORY_NAME = __file__.split('/')[-1].rstrip('.py')
DEFAULT_SAVE_DIRECTORY_NAME += time.strftime('_%mm_%dd_%Hh_%Mm_%Ss', time.localtime())
DEFAULT_SAVE_DIRECTORY = os.path.join('save/', DEFAULT_SAVE_DIRECTORY_NAME)


@configurable
def main(dataset=reverse,
         checkpoint_path=None,
         save_directory=DEFAULT_SAVE_DIRECTORY,
         device=None,
         random_seed=None,
         epochs=4,
         train_batch_size=16,
         dev_batch_size=128):
    ###############################################################################
    # Subscribe handler to root logger
    ###############################################################################
    handler = logging.FileHandler(os.path.join(save_directory, 'train.log'))
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logger.handlers[0].formatter)
    logger.addHandler(handler)

    ###############################################################################
    # Configure hyperparameters
    ###############################################################################
    add_config(DEFAULT_HYPERPARAMETERS)

    ###############################################################################
    # Pandas
    ###############################################################################
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('max_colwidth', 80)

    ###############################################################################
    # Setup Device
    ###############################################################################
    device = device_default(device)
    if device is not None and torch.cuda.is_available():
        torch.cuda.set_device(device)

    logger.info('Device: %s', device)

    ###############################################################################
    # Random Seed
    ###############################################################################
    if random_seed:
        random_seed = int(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    ###############################################################################
    # Load Checkpoint
    ###############################################################################
    checkpoint = Checkpoint.load(
        checkpoint_path=checkpoint_path, save_directory=save_directory, device=device)

    ###############################################################################
    # Init Dataset
    ###############################################################################
    train_dataset, dev_dataset = dataset(train=True, dev=True, test=False)
    # NOTE: `source` and `target` are typically associated with sequences
    assert 'source' in train_dataset
    assert 'target' in train_dataset

    logger.info('Num Training Data: %d', len(train_dataset))
    logger.info('Num Development Data: %d', len(dev_dataset))

    ###############################################################################
    # Init Encoders
    ###############################################################################
    if checkpoint:
        source_encoder = checkpoint.input_encoder
        target_encoder = checkpoint.output_encoder
    else:
        source_encoder = WordEncoder(train_dataset['source'])
        target_encoder = WordEncoder(train_dataset['target'])

    ###############################################################################
    # Encode Dataset
    ###############################################################################
    for dataset in [train_dataset, dev_dataset]:
        for row in dataset:
            row['source'] = source_encoder.encode(row['source'])
            row['target'] = target_encoder.encode(row['target'])

    ###############################################################################
    # Init Model
    ###############################################################################
    if checkpoint:
        model = checkpoint.model
    else:
        model = Seq2seq(
            EncoderRNN(vocab_size=source_encoder.vocab_size, embeddings=source_encoder.embeddings),
            DecoderRNN(vocab_size=target_encoder.vocab_size, embeddings=target_encoder.embeddings))
        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)

    logger.info('Model:\n%s', model)
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0]
                       for x in model.parameters())
    logger.info('Total Parameters: %d', total_params)

    ###############################################################################
    # Init Loss
    ###############################################################################
    criterion = NLLLoss(ignore_index=PADDING_INDEX)

    ###############################################################################
    # Move the model and loss to CUDA memory
    ###############################################################################
    if torch.cuda.is_available():
        model.cuda(device_id=device)
        criterion.cuda()

    ###############################################################################
    # Init Optimizer
    ###############################################################################
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Optimizer(Adam(params=params))
    optimizer.set_scheduler(StepLR)

    ###############################################################################
    # Init Observers
    ###############################################################################
    dev_observers = [
        Accuracy(ignore_index=PADDING_INDEX), RandomSample(
            source_encoder, target_encoder, n_samples=5, ignore_index=PADDING_INDEX)
    ]

    ###############################################################################
    # Start Training
    ###############################################################################
    collate_fn_partial = partial(
        collate_fn, input_key='source', output_key='target', sort_key='source')

    for i in range(epochs):
        logger.info('Starting epoch %d', i)
        model.train(mode=True)
        for batch in tqdm(
                DataLoader(
                    train_dataset,
                    batch_sampler=BatchSamplerShuffle(
                        SortedSampler(train_dataset, lambda r: r['source'].size()[0]),
                        train_batch_size, False),
                    collate_fn=collate_fn_partial,
                    pin_memory=torch.cuda.is_available(),
                    num_workers=0)):
            source, source_lengths = batch['source']
            target, target_lengths = batch['target']

            if torch.cuda.is_available():
                source, source_lengths = source.cuda(async=True), source_lengths.cuda(async=True)
                target, target_lengths = target.cuda(async=True), target_lengths.cuda(async=True)
            source = Variable(source)
            target = Variable(target)

            optimizer.zero_grad()
            output = model(source, source_lengths, target, target_lengths)[0]

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
        for batch in DataLoader(
                dev_dataset,
                batch_size=dev_batch_size,
                collate_fn=collate_fn_partial,
                pin_memory=torch.cuda.is_available(),
                num_workers=0):
            source, source_lengths = batch['source']
            target, target_lengths = batch['target']
            if torch.cuda.is_available():
                source, source_lengths = source.cuda(async=True), source_lengths.cuda(async=True)
                target, target_lengths = target.cuda(async=True), target_lengths.cuda(async=True)
            source = Variable(source)
            target = Variable(target)
            output = model(source, source_lengths, target, target_lengths)[0]
            [
                observer.update({
                    'source_batch': source,
                    'target_batch': target,
                    'output_batch': output
                }) for observer in dev_observers
            ]

        [observer.dump().reset() for observer in dev_observers]

    # 8. Return the best loss function

    # If any of these steps seems repetitive create a function to help that in training?


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
