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
from tqdm import tqdm

import torch

from lib.checkpoint import Checkpoint
from lib.configurable import add_config
from lib.configurable import configurable
from lib.datasets import reverse
from lib.nn import DecoderRNN
from lib.nn import EncoderRNN
from lib.nn import Seq2seq
from lib.observers import Accuracy
from lib.optim import Optimizer
from lib.text_encoders import PADDING_INDEX
from lib.text_encoders import WordEncoder
from lib.utils import device_default
from lib.utils import init_logging
from lib.utils import collate_fn

Adam.__init__ = configurable(Adam.__init__)
StepLR.__init__ = configurable(StepLR.__init__)

# NOTE: The goal of this file is just to setup the training for simple_questions_predicate.

BASE_RNN_HYPERPARAMETERS = {
    'embedding_size': 300,
    'rnn_size': 256,
    'n_layers': 2,
    'rnn_cell': 'lstm',
    'embedding_dropout': 0.4,
    'rnn_variational_dropout': 0.0,
}

DEFAULT_HYPERPARAMETERS = {
    'scripts.simple_questions_predicate': {
        'train_batch_size': 32,
        'epochs': 10,
    },
    'lib': {
        'nn': {
            'decoder_rnn.DecoderRNN.__init__': {
                'rnn_dropout': 0.0,
                'use_attention': True,
            },
            'encoder_rnn.EncoderRNN.__init__': {
                'rnn_dropout': 0.25,
                'bidirectional': True,
                'freeze_embeddings': False,
            },
            'attention.Attention.__init__.attention_type': 'general',
        },
        'optim.optim.Optimizer.__init__': {
            'max_grad_norm': 0.65,
        }
    },
    'torch.optim.Adam.__init__': {
        'lr': 0.001584,
        'weight_decay': 0,
    },
    'torch.optim.lr_scheduler.StepLR.__init__': {
        'step_size': 5
    }
}

DEFAULT_HYPERPARAMETERS['lib']['nn']['decoder_rnn.DecoderRNN.__init__'].update(
    BASE_RNN_HYPERPARAMETERS)
DEFAULT_HYPERPARAMETERS['lib']['nn']['encoder_rnn.EncoderRNN.__init__'].update(
    BASE_RNN_HYPERPARAMETERS)

add_config(DEFAULT_HYPERPARAMETERS)

DEFAULT_SAVE_DIRECTORY_NAME = __file__.split('/')[-1].rstrip('.py')
DEFAULT_SAVE_DIRECTORY_NAME += time.strftime('_%mm_%dd_%Hh_%Mm_%Ss', time.localtime())
DEFAULT_SAVE_DIRECTORY = os.path.join('save/', DEFAULT_SAVE_DIRECTORY_NAME)

logger = logging.getLogger(__name__)


@configurable
def main(dataset=reverse,
         checkpoint_path=None,
         save_directory=DEFAULT_SAVE_DIRECTORY,
         device=None,
         random_seed=None,
         epochs=10,
         train_batch_size=32,
         dev_batch_size=128):
    ###############################################################################
    # Logging
    ###############################################################################
    init_logging(save_directory)

    ###############################################################################
    # Setup Device
    ###############################################################################
    device = device_default(device)
    if device is not None and torch.cuda.is_available():
        torch.cuda.set_device(device)

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
    # Init Observers
    ###############################################################################
    train_observers = []  # [Accuracy(ignore_index=PADDING_INDEX)]
    dev_observers = []  # [Accuracy(ignore_index=PADDING_INDEX)]

    ###############################################################################
    # Init Dataset
    ###############################################################################
    train_dataset, dev_dataset = dataset(train=True, dev=True, test=False)
    # NOTE: `source` and `target` are typically associated with sequences
    assert 'source' in train_dataset
    assert 'target' in train_dataset

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
    # Start Training
    ###############################################################################
    collate_fn_partial = partial(
        collate_fn, input_key='source', output_key='target', sort_key='source')

    for i in range(epochs):
        model.train(mode=True)
        for batch in tqdm(
                DataLoader(
                    train_dataset,
                    batch_size=train_batch_size,
                    shuffle=True,
                    collate_fn=collate_fn_partial)):
            source, source_lengths = batch['source']
            target, target_lengths = batch['target']
            output = model(source, source_lengths, target, target_lengths)[0]

            [observer.update(batch, output) for observer in train_observers]

            # Compute loss
            # NOTE: flattening the tensors allows for the computation to be done once per batch
            output_flat = output.view(-1, output.size(2))
            target_flat = target.view(-1)
            loss = criterion(output_flat, target_flat)

            # Backward propagation
            model.zero_grad()
            loss.backward()
            optimizer.step()

        [observer.reset() for observer in train_observers]
        print('here')
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
                dev_dataset, batch_size=dev_batch_size, collate_fn=collate_fn_partial):
            source, source_lengths = batch['source']
            target, target_lengths = batch['target']
            output = model(source, source_lengths, target, target_lengths)[0]
            [observer.update(batch, output) for observer in dev_observers]
        [observer.reset() for observer in dev_observers]

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
    main(checkpoint_path=args.checkpoint_path, save_directory=save_directory)
