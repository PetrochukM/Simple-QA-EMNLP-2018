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
from lib.configurable import configurable
from lib.datasets import reverse
from lib.metrics import get_accuracy
from lib.metrics import get_bucket_accuracy
from lib.metrics import get_random_sample
from lib.metrics import get_token_accuracy
from lib.nn import DecoderRNN
from lib.nn import EncoderRNN
from lib.nn import SeqToSeq
from lib.optim import Optimizer
from lib.samplers import BucketBatchSampler
from lib.text_encoders import PADDING_INDEX
from lib.text_encoders import WordEncoder
from lib.utils import collate_fn
from lib.utils import get_total_parameters
from lib.utils import init_logging
from lib.utils import setup_training

init_logging()
logger = logging.getLogger(__name__)  # Root logger

Adam.__init__ = configurable(Adam.__init__)
StepLR.__init__ = configurable(StepLR.__init__)

pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 80)

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

DEFAULT_SAVE_DIRECTORY_NAME = __file__.split('/')[-1].replace('.py', '')
DEFAULT_SAVE_DIRECTORY_NAME += time.strftime('_%mm_%dd_%Hh_%Mm_%Ss', time.localtime())
DEFAULT_SAVE_DIRECTORY = os.path.join('save/', DEFAULT_SAVE_DIRECTORY_NAME)


def train(dataset=reverse,
          checkpoint_path=None,
          save_directory=DEFAULT_SAVE_DIRECTORY,
          hyperparameters_config=DEFAULT_HYPERPARAMETERS,
          device=None,
          random_seed=123,
          epochs=4,
          train_max_batch_size=16,
          dev_max_batch_size=128):
    checkpoint = setup_training(dataset, checkpoint_path, save_directory, hyperparameters_config,
                                device, random_seed)

    # Init Dataset
    train_dataset, dev_dataset = dataset(train=True, dev=True, test=False)

    logger.info('Num Training Data: %d', len(train_dataset))
    logger.info('Num Development Data: %d', len(dev_dataset))

    # Init Encoders and encode dataset
    if checkpoint:
        source_encoder, target_encoder = checkpoint.input_encoder, checkpoint.output_encoder
    else:
        source_encoder = WordEncoder(train_dataset['source'])
        target_encoder = WordEncoder(train_dataset['target'])
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
        return Variable(source), source_lengths, Variable(target), target_lengths

    # Train!
    for i in range(epochs):
        logger.info('Epoch %d', i)
        model.train(mode=True)
        batch_sampler = BucketBatchSampler(train_dataset, lambda r: r['source'].size()[0],
                                           train_max_batch_size)
        train_iterator = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn_partial,
            pin_memory=torch.cuda.is_available(),
            num_workers=0)
        for batch in tqdm(train_iterator):
            source, source_lengths, target, target_lengths = prepare_batch(batch)

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

        # Evaluate
        model.train(mode=False)
        outputs, sources, targets = [], [], []
        dev_iterator = DataLoader(
            dev_dataset,
            batch_size=dev_max_batch_size,
            collate_fn=collate_fn_partial,
            pin_memory=torch.cuda.is_available(),
            num_workers=0)
        for batch in dev_iterator:
            source, source_lengths, target, target_lengths = prepare_batch(batch)
            output = model(source, source_lengths, target, target_lengths)[0]
            # Prevent memory leak by moving output from variable to tensor
            sources.extend(source.data.cpu().transpose(0, 1).split(split_size=1, dim=0))
            targets.extend(target.data.cpu().transpose(0, 1).split(split_size=1, dim=0))
            outputs.extend(output.data.cpu().transpose(0, 1).split(split_size=1, dim=0))

        get_accuracy(targets, outputs, ignore_index=PADDING_INDEX, print=True)
        get_token_accuracy(targets, outputs, ignore_index=PADDING_INDEX, print=True)
        buckets = [t.ne(PADDING_INDEX).sum() - 1 for t in targets]
        get_bucket_accuracy(buckets, targets, outputs, ignore_index=PADDING_INDEX, print=True)
        get_random_sample(
            sources,
            targets,
            outputs,
            source_encoder,
            target_encoder,
            ignore_index=PADDING_INDEX,
            print=True)

    # TODO: Return the best loss if hyperparameter tunning.

    # TODO: Figure out a good abstraction for evaluation.
    # TODO: In class, I'll add a classification model and try to get it running.


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
    train(checkpoint_path=args.checkpoint_path, save_directory=save_directory)
