from functools import partial

import argparse
import logging
import os

from torch.nn.modules.loss import NLLLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

import torch
import pandas as pd

from lib.checkpoint import Checkpoint
from lib.configurable import add_config
from lib.configurable import configurable
from lib.datasets import simple_qa_predicate_preprocessed
from lib.datasets import simple_qa_predicate
from lib.metrics import get_accuracy
from lib.metrics import get_accuracy_top_k
from lib.metrics import print_bucket_accuracy
from lib.metrics import print_random_sample
from lib.nn import SeqToLabel
from lib.optim import Optimizer
from lib.pretrained_embeddings import FastText
from lib.samplers import BucketBatchSampler
from lib.samplers import SortedSampler
from lib.text_encoders import IdentityEncoder
from lib.text_encoders import TreebankEncoder
from lib.text_encoders import WordEncoder
from lib.utils import get_log_directory_path
from lib.utils import get_total_parameters
from lib.utils import init_logging
from lib.utils import pad_batch
from lib.utils import setup_training

Adam.__init__ = configurable(Adam.__init__)

# TODO: Train this in Notebook

pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 80)

embedding_size = 300
unk_init = lambda t: torch.FloatTensor(t).uniform_(-0.1, 0.1)

SIMPLE_PREDICATE_HYPERPARAMETERS = {
    'lib': {
        'nn.seq_to_label.SeqToLabel.__init__': {
            'bidirectional': True,
            'embedding_size': embedding_size,
            'rnn_size': 300,
            'freeze_embeddings': True,
            'rnn_cell': 'lstm',
            'decode_dropout': 0.3,  # dropout before fully connected layer in RNN
            'rnn_layers': 2,
            'rnn_variational_dropout': 0.3,
            'embedding_dropout': 0.3
        },
        'optim.Optimizer.__init__.max_grad_norm': None,
    },
    'torch.optim.adam.Adam.__init__': {
        'lr': 1e-4,
        'weight_decay': 0,
    },
    'scripts.python.train_relation_classifier.train': {
        'get_dataset': lambda: simple_qa_predicate,
        'random_seed': 3435,
        'epochs': 30,
        'train_max_batch_size': 16,
        'dev_max_batch_size': 128,
        'get_pretrained_embedding': lambda: FastText(language="en"),
    }
}

add_config(SIMPLE_PREDICATE_HYPERPARAMETERS)


@configurable
def train(
        log_directory,  # Logs experiments, checkpoints, etc are saved
        get_dataset,
        random_seed,
        epochs,
        train_max_batch_size,
        dev_max_batch_size,
        get_pretrained_embedding=None,
        checkpoint_path=None,
        device=None):
    is_cuda, checkpoint = setup_training(checkpoint_path, log_directory, device,
                                         random_seed)  # Async minibatch allocation for speed
    # Reference: http://timdettmers.com/2015/03/09/deep-learning-hardware-guide/
    # TODO: look into cuda_async device=device
    cuda_async = lambda t: t.cuda(async=True) if is_cuda else t  # Use with tensors
    cuda = lambda t: t.cuda(device_id=device) if is_cuda else t  # Use with nn.modules

    # Init Dataset
    dataset = get_dataset()
    train_dataset, dev_dataset = dataset(train=True, dev=True, test=False)
    logger.info('Num Training Data: %d', len(train_dataset))
    logger.info('Num Development Data: %d', len(dev_dataset))

    # Init Encoders and encode dataset
    if checkpoint:
        text_encoder, relation_encoder = checkpoint.input_encoder, checkpoint.output_encoder
    else:
        text_encoder = TreebankEncoder(
            train_dataset['text'] + dev_dataset['text'], lower=True, append_eos=False)
        logger.info('Text encoder vocab size: %d' % text_encoder.vocab_size)
        relation_encoder = IdentityEncoder(train_dataset['relation'] + dev_dataset['relation'])
        logger.info('Relation encoder vocab size: %d' % relation_encoder.vocab_size)
    for dataset in [train_dataset, dev_dataset]:
        for row in dataset:
            row['text'] = text_encoder.encode(row['text'])
            row['relation'] = relation_encoder.encode(row['relation'])

    # Init Model
    if checkpoint:
        model = checkpoint.model
    else:
        model = SeqToLabel(text_encoder.vocab_size, relation_encoder.vocab_size)
        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)

        # Load embeddings
        if get_pretrained_embedding:
            pretrained_embedding = get_pretrained_embedding()
            embedding_weights = torch.Tensor(text_encoder.vocab_size, pretrained_embedding.dim)
            for i, token in enumerate(text_encoder.vocab):
                embedding_weights[i] = pretrained_embedding[token]
            model.encoder.embedding.weight.data.copy_(embedding_weights)

    cuda(model)

    logger.info('Model:\n%s', model)
    logger.info('Total Parameters: %d', get_total_parameters(model))

    # Init Loss
    criterion = cuda(NLLLoss())

    # Init Optimizer
    # https://github.com/pytorch/pytorch/issues/679
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Optimizer(Adam(params=params))

    def collate_fn(batch, train=True):
        """ list of tensors to a batch variable """
        # PyTorch RNN requires sorting decreasing size
        batch = sorted(batch, key=lambda row: len(row['text']), reverse=True)
        input_batch, input_lengths = pad_batch([row['text'] for row in batch])
        relations = [row['relation'] for row in batch]

        # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
        to_variable = (
            lambda b: Variable(torch.stack(b).t_().squeeze(0).contiguous(), volatile=not train))

        return (to_variable(input_batch), torch.LongTensor(input_lengths), to_variable(relations))

    # Train!
    sort_key = lambda r: r['text'].size()[0]
    for epoch in range(epochs):
        logger.info('Epoch %d', epoch)
        model.train(mode=True)
        batch_sampler = BucketBatchSampler(train_dataset, sort_key, train_max_batch_size)
        train_iterator = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            pin_memory=is_cuda,
            num_workers=0)
        for text, text_lengths, relation in tqdm(train_iterator):
            optimizer.zero_grad()
            relation = cuda_async(relation)
            output = model(cuda_async(text), cuda_async(text_lengths))
            loss = criterion(output, relation)

            # Backward propagation
            loss.backward()
            optimizer.step()

        Checkpoint.save(
            log_directory=log_directory,
            model=model,
            optimizer=optimizer,
            input_text_encoder=text_encoder,
            output_text_encoder=relation_encoder,
            device=device)

        # Evaluate
        model.train(mode=False)
        texts, relations, outputs = [], [], []
        dev_iterator = DataLoader(
            dev_dataset,
            batch_size=dev_max_batch_size,
            sampler=SortedSampler(dev_dataset, sort_key, sort_noise=0.0),
            collate_fn=partial(collate_fn, train=False),
            pin_memory=is_cuda,
            num_workers=0)
        total_loss = 0
        for text, text_lengths, relation in dev_iterator:
            relation = cuda_async(relation)
            output = model(cuda_async(text), cuda_async(text_lengths))
            total_loss += criterion(output, relation).data[0] * relation.size()[0]
            # Prevent memory leak by moving output from variable to tensor
            texts.extend(text.data.cpu().transpose(0, 1).split(split_size=1, dim=0))
            relations.extend(relation.data.cpu().split(split_size=1, dim=0))
            outputs.extend(output.data.cpu().split(split_size=1, dim=0))

        optimizer.update(total_loss / len(dev_dataset), epoch)
        print_random_sample(texts, relations, outputs, text_encoder, relation_encoder, n_samples=5)
        # buckets = [relation_encoder.decode(relation) for relation in relations]
        # print_bucket_accuracy(buckets, relations, outputs)
        logger.info('Loss: %.03f', total_loss / len(dev_dataset))
        get_accuracy(relations, outputs, print_=True)
        get_accuracy_top_k(relations, outputs, k=3, print_=True)
        get_accuracy_top_k(relations, outputs, k=5, print_=True)
        print()

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

    if args.checkpoint_path:
        log_directory = os.path.dirname(args.checkpoint_path)
    else:
        log_directory = get_log_directory_path('relation_classifier')
    log_directory = init_logging(log_directory)
    logger = logging.getLogger(__name__)
    train(checkpoint_path=args.checkpoint_path, log_directory=log_directory)