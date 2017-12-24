from functools import partial
from collections import defaultdict

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
from lib.datasets import simple_qa_predicate
from lib.metrics import get_accuracy
from lib.metrics import print_bucket_accuracy
from lib.metrics import print_random_sample
from lib.nn import SeqToSeq
from lib.optim import Optimizer
from lib.pretrained_embeddings import GloVe
from lib.samplers import BucketBatchSampler
from lib.samplers import SortedSampler
from lib.text_encoders import TreebankEncoder
from lib.text_encoders import WordEncoder
from lib.text_encoders import PADDING_INDEX
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

SIMPLE_PREDICATE_HYPERPARAMETERS = {
    'lib': {
        'nn.seq_to_seq.SeqToSeq.__init__': {
            'embedding_size': embedding_size,
            'rnn_size': 256,
            'n_layers': 2,
            'rnn_cell': 'lstm',
            'tie_weights': False
        },
        'nn.seq_encoder.SeqEncoder.__init__': {
            'embedding_dropout': 0.2,
            'rnn_dropout': 0.25,
            'bidirectional': True,
            'freeze_embeddings': True,
        },
        'nn.seq_decoder.SeqDecoder.__init__': {
            'embedding_dropout': 0.4,
            'use_attention': True,
            'rnn_dropout': 0.0,
            'freeze_embeddings': False,
            'decode_dropout': 0.3,
        },
        'optim.Optimizer.__init__.max_grad_norm': 1.0,
        'attention.Attention.__init__.attention_type': 'general',
    },
    'torch.optim.adam.Adam.__init__': {
        'lr': 0.001584,
        'weight_decay': 0,
    },
    'scripts.python.train_relation_seq_to_seq.train': {
        'get_dataset':
            lambda: simple_qa_predicate,
        'random_seed':
            3435,
        'epochs':
            30,
        'train_max_batch_size':
            16,
        'dev_max_batch_size':
            128,
        'get_pretrained_embedding':
            lambda **kwargs: GloVe(name='6B', dim=str(embedding_size), **kwargs),
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
    relation_tree = {}
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

    # Preprocess dataset
    for dataset in [train_dataset, dev_dataset]:
        for row in dataset:
            row['relation'] = row['relation'].replace('www.freebase.com/', '')
            row['relation'] = row['relation'].replace('_', ' _ ')
            row['relation'] = row['relation'].replace('/', ' / ')

    # Init Encoders and encode dataset
    if checkpoint:
        text_encoder, relation_encoder = checkpoint.input_encoder, checkpoint.output_encoder
    else:
        text_encoder = TreebankEncoder(train_dataset['text'] + dev_dataset['text'], lower=True)
        logger.info('Text encoder vocab size: %d' % text_encoder.vocab_size)
        relation_encoder = WordEncoder(
            train_dataset['relation'] + dev_dataset['relation'], append_eos=True)
        logger.info('Relation encoder vocab size: %d' % relation_encoder.vocab_size)

    for dataset in [train_dataset, dev_dataset]:
        for row in dataset:
            row['text'] = text_encoder.encode(row['text'])
            row['relation'] = relation_encoder.encode(row['relation'])

            # Build up a tree of all possible relations
            tree_node = relation_tree
            for index in row['relation']:
                if index not in tree_node:
                    tree_node[index] = {}
                tree_node = tree_node[index]

    # Using the tree, check if a particular sequence is possible
    def include_vocab(predictions):
        tree_node = relation_tree
        for prediction in predictions:
            if prediction not in tree_node:
                return None
            tree_node = tree_node[prediction]
        return list(tree_node.keys())

    # Init Model
    if checkpoint:
        model = checkpoint.model
    else:
        model = SeqToSeq(
            text_encoder.vocab_size, relation_encoder.vocab_size, include_vocab=include_vocab)

        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)

        # Load embeddings
        if get_pretrained_embedding:
            unk_init = lambda t: torch.FloatTensor(t).uniform_(-0.1, 0.1)
            pretrained_embedding = get_pretrained_embedding(unk_init=unk_init)
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
        output_batch, output_lengths = pad_batch([row['relation'] for row in batch])

        # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
        to_variable = (
            lambda b: Variable(torch.stack(b).t_().squeeze(0).contiguous(), volatile=not train))

        return (to_variable(input_batch), torch.LongTensor(input_lengths),
                to_variable(output_batch), torch.LongTensor(output_lengths))

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
        for text, text_lengths, relation, relation_lengths in tqdm(train_iterator):
            optimizer.zero_grad()
            relation = cuda_async(relation)
            output = model(
                cuda_async(text), cuda_async(text_lengths), relation,
                cuda_async(relation_lengths))[0]

            # Compute loss
            # NOTE: flattening the tensors allows for the computation to be done once per batch
            output_flat = output.view(-1, output.size(2))
            relation_flat = relation.view(-1)
            loss = criterion(output_flat, relation_flat)

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
        for text, text_lengths, relation, _ in tqdm(dev_iterator):
            output = model(cuda_async(text), cuda_async(text_lengths))[0]

            # Output could of stopped early or late;
            # therefore, it is not same sequence length as relation
            relation = cuda_async(relation)
            if output.size()[0] > relation.size()[0]:
                output = output[:relation.size()[0]]
            elif output.size()[0] < relation.size()[0]:
                relation = relation[:output.size()[0]]

            # Compute loss
            # NOTE: flattening the tensors allows for the computation to be done once per batch
            output_flat = output.view(-1, output.size(2))
            relation_flat = relation.view(-1)
            total_loss += criterion(output_flat, relation_flat).data[0] * relation_flat.size()[0]

            # Prevent memory leak by moving output from variable to tensor
            texts.extend(text.data.cpu().transpose(0, 1).split(split_size=1, dim=0))
            relations.extend(relation.data.cpu().transpose(0, 1).split(split_size=1, dim=0))
            outputs.extend(output.data.cpu().transpose(0, 1).split(split_size=1, dim=0))

        optimizer.update(total_loss / len(dev_dataset), epoch)
        print_random_sample(texts, relations, outputs, text_encoder, relation_encoder, n_samples=5)
        # buckets = [relation_encoder.decode(relation) for relation in relations]
        # print_bucket_accuracy(buckets, relations, outputs)
        logger.info('Loss: %.03f', total_loss / len(dev_dataset))
        get_accuracy(relations, outputs, print_=True)

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
        log_directory = get_log_directory_path('relation_seq_to_seq')
    log_directory = init_logging(log_directory)
    logger = logging.getLogger(__name__)
    train(checkpoint_path=args.checkpoint_path, log_directory=log_directory)
