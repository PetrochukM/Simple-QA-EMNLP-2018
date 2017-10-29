# TODO: The goal of the training script is not to have a reusable procedure. Just reusable functions.
# TODO: Not end to end resuable.
# Create some state and change it
"""
Executable to train on a configuration file.

TODO: Support getting the checkpoint that did best on the development set in the latest experiment.
TODO: Set a bag of words baseline.
TODO: Add curriculum learning.
TODO: Checkpoints should have client names attached and store more useful information.
TODO: Vocab between shared source and target should be not generated twice.
TODO: Implement copy attention.
TODO: Implement coverage attention.
"""
import argparse
import logging
import yaml
import os

import torch
import pandas as pd

from utils.config.config import get_root_path
from utils.config.config import init_logging
from utils.config.configurable import add_config
from utils.config.configurable import configurable
from utils.controllers import Evaluator
from utils.controllers import Trainer
from utils.datasets.utils import get_dataset
from utils.fields import SeqField
from utils.metrics.utils import get_loss
from utils.metrics.utils import get_evaluate_loss
from utils.metrics.utils import get_metrics
from utils.models import DecoderRNN
from utils.models import EncoderRNN
from utils.models import Seq2seq
from utils.optim.optim import Optimizer
from utils.checkpoint import save, load


###############################################################################
# Logging
###############################################################################
init_logging()
logger = logging.getLogger(__name__)


###############################################################################
# Pandas
###############################################################################
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 80)


###############################################################################
# Command line 
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Configuration to load for default parameters.')
parser.add_argument(
    '--load_checkpoint',
    const=True,
    default=False,
    action='store',
    nargs='?',
    type=str,
    help="""Indicates if training has to be resumed from the latest checkpoint or the name of
            the checkpoint to load.""")
args = parser.parse_args()


###############################################################################
# Global Configuration
###############################################################################
config = yaml.load(open(os.path.join(get_root_path(), args.config)))
add_config(config)


###############################################################################
# Init model, input_field, output_field.
###############################################################################
if args.load_checkpoint:
    pass
else:
    pass


###############################################################################
# Load dataset, Update input_field, output_field
###############################################################################
# Load the dataset with input_field, and output_field
if args.load_checkpoint:
    pass
else:
    pass

# Two routes: Have a checkpoint or not
# Checkpoint saves everything required to run the model


# TODO: How do we imagine the design of this?
# Much like the PyTorch model, we create the input field and the output field


if checkpoint_name else None

checkpoint = (
    Checkpoint.get_checkpoint(checkpoint_name=checkpoint_name) if checkpoint_name else None)
train_data, development_data, input_field, output_field = get_dataset(
    load_test=False, checkpoint=checkpoint)


@configurable
def train(train_data,
          development_data,
          input_field,
          output_field,
          inference_metrics=[],
          train_metrics=[],
          device=None,
          checkpoint=None):
    """
    Runs the trainer with `train_data` and `development_data`.

    Args:
        train_data (torchtext.data.Dataset): dataset holding a list of examples for training
        development_data (torchtext.data.Dataset): dataset holding a list of examples for development
        input_field (torchtext.data.Field): input field for processing the source input
        output_field (torchtext.data.Field): output field for processing the target output
        inference_metrics (list of seq2seq.metrics.Metric): list of metrics to evaluate on the test
            set
        train_metrics (list of seq2seq.metrics.Metric): list of metrics to evaluate on the train
            data
        device (int or None): -1 for CPU, None for default GPU, and 0+ for GPU device ID
        checkpoint (seq2seq.util.Checkpoint): Checkpoint to load
    Returns:
        (float) min loss on the development_data
    """
    # Prepare iterators and loss
    logger.info('Training...')
    logger.info('Training Data: %d', len(train_data))
    logger.info('Development Data: %d', len(development_data))
    logger.info('Device: %s', device)

    if device is not None and torch.cuda.is_available():
        torch.cuda.set_device(device)

    padding_idx = output_field.vocab.stoi[SeqField.PAD_TOKEN]

    loss = get_loss(ignore_index=padding_idx)

    # Set up evaluator
    evaluate_loss = get_evaluate_loss(
        ignore_index=padding_idx,
        input_field=input_field,
        output_field=output_field,
        mask=padding_idx)
    inference_metrics = get_metrics(
        inference_metrics, input_field=input_field, output_field=output_field, mask=padding_idx)
    evaluator = Evaluator(metrics=inference_metrics, loss=evaluate_loss)

    optimizer = Optimizer()

    # Set up trainer
    train_metrics = get_metrics(
        train_metrics, output_field=output_field, mask=padding_idx, input_field=input_field)
    trainer = Trainer(
        optimizer=optimizer, loss=loss, device=device, evaluator=evaluator, metrics=train_metrics)

    # Set up model
    if checkpoint:
        model = checkpoint.model
    else:
        model = Seq2seq(EncoderRNN(input_field.vocab), DecoderRNN(output_field.vocab))
        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)

    optimizer.set_parameters(model.parameters())

    if checkpoint:
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)

    # Move from CPU to GPU memory
    if torch.cuda.is_available():
        if device is not None and device > 0:
            torch.cuda.set_device(device)
        model.cuda(device_id=device)
        loss.cuda()

    logger.info('Model:\n%s', model)
    total_params = sum(
        x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
    logger.info('Total Parameters: %d', total_params)

    return trainer.train(model, train_data, development_data)


@configurable
def main(config, checkpoint_name=None):
    """ Given a configuration and optional checkpoint, load data, preprocess, and train.

    Args:
        config (dict): dictionary to add a configuration
        checkpoint_name (str): name of the checkpoint to load or None for the most recent
    Returns:
        (float) min loss on the development_data
    """
    add_config(config)
    checkpoint = (
        Checkpoint.get_checkpoint(checkpoint_name=checkpoint_name) if checkpoint_name else None)
    train_data, development_data, input_field, output_field = get_dataset(
        load_test=False, checkpoint=checkpoint)
    return train(train_data, development_data, input_field, output_field, checkpoint=checkpoint)


if __name__ == '__main__':  # pragma: no cover
    config = yaml.load(open(os.path.join(get_root_path(), options.config)))
    main(config, options.load_checkpoint)
