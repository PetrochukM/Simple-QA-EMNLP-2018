# coding: utf-8

# # Yu et al. 2017 Relation Model
#
# Our goal here to to reimplement Yu et al. 2017 93% relation model.
#
# First things first, set up the initial configuration.

# In[1]:

import sys
print('Python Version:', sys.version)
import pandas as pd
import logging

from lib.utils import setup_training

# Create root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 80)

random_seed = 123
device = 0
is_cuda, _ = setup_training(device, random_seed)
# Async minibatch allocation for speed
# Reference: http://timdettmers.com/2015/03/09/deep-learning-hardware-guide/
# TODO: look into cuda_async device=device
cuda_async = lambda t: t.cuda(device=device, async=True) if is_cuda else t  # Use with tensors
cuda = lambda t: t.cuda(device=device) if is_cuda else t  # Use with nn.modules

# ## Dataset
#
# Load our dataset. Log a couple rows.

# In[2]:

import os
from tqdm import tqdm

from lib.datasets.dataset import Dataset


def yu_dataset(directory='data/yu/',
               train=False,
               dev=False,
               test=False,
               train_filename='train.replace_ne.withpool',
               dev_filename='valid.replace_ne.withpool',
               test_filename='test.replace_ne.withpool',
               vocab_filename='relation.2M.list'):
    """
    Example line example: 40\t61 40 117\twhich genre of album is #head_entity# ?
    Vocab example: /film/film/genre
    
    Sample Data:
        Question: 'which genre of album is #head_entity# ?'
        True Relation: '/music/album/genre'
        False Relation Pool: ['/music/album/release_type', '/music/album/genre', '/music/album/artist']
    """
    vocab_path = os.path.join(directory, vocab_filename)
    vocab = [l.strip() for l in open(vocab_path, 'r')]

    ret = []
    datasets = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    for is_requested, filename in datasets:
        if not is_requested:
            continue

        file_path = os.path.join(directory, filename)
        data = pd.read_table(
            file_path, header=None, names=['True Relation', 'Relation Pool', 'Question'])
        rows = []
        for i, row in tqdm(data.iterrows(), total=data.shape[0]):
            if row['Relation Pool'].strip() == 'noNegativeAnswer':
                continue
            relation_pool = [vocab[int(i) - 1] for i in row['Relation Pool'].split()]
            true_relation = vocab[int(row['True Relation']) - 1]
            question = row['Question'].strip()
            # Development and test set may or may not have the True relation based on our predicted pool
            if filename == train_filename:
                assert true_relation not in relation_pool

            for relation in relation_pool:
                if filename == train_filename:
                    rows.append({
                        'Question': question,
                        'True Relation': true_relation,
                        'False Relation': relation,
                        'Example ID': i
                    })
                else:
                    rows.append({
                        'Question': question,
                        'True Relation': true_relation,
                        'Relation': relation,
                        'Example ID': i
                    })
        ret.append(Dataset(rows))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


# In[3]:

from IPython.display import display

train_dataset, dev_dataset = yu_dataset(train=True, dev=True)

print('Num Training Data: %d' % len(train_dataset))
print('Train Sample:')
display(pd.DataFrame(train_dataset[:5]))
print('\nNum Development Data: %d' % len(dev_dataset))
print('Development Sample:')
display(pd.DataFrame(dev_dataset[:5]))

# ## Encode Text
#
# Here we encode our data into a numerical format.

# In[4]:

from IPython.display import display
import re

from lib.text_encoders import StaticTokenizerEncoder
from lib.text_encoders import DelimiterEncoder
from lib.text_encoders import WordEncoder

# We add development dataset to text_encoder for embeddings
# We make sure not to use the the development dataset to provide us with any vocab optimizations or learning
text_encoder = WordEncoder(
    train_dataset['Question'] + dev_dataset['Question'], lower=True, append_eos=False)
print('Text encoder vocab size: %d' % text_encoder.vocab_size)

relations = set(train_dataset['True Relation'] + train_dataset['False Relation'])
relation_word_encoder = StaticTokenizerEncoder(relations, tokenize=lambda s: re.split('/|_', s))
print('Relation word encoder vocab size: %d' % relation_word_encoder.vocab_size)

relation_encoder = DelimiterEncoder('/', relations)
print('Relation encoder vocab size: %d' % relation_encoder.vocab_size)

for dataset in [train_dataset, dev_dataset]:
    for row in dataset:
        row['Question'] = text_encoder.encode(row['Question'])
        row['True Relation Word'] = relation_word_encoder.encode(row['True Relation'])
        row['True Relation'] = relation_encoder.encode(row['True Relation'])

        if 'False Relation' in row:
            row['False Relation Word'] = relation_word_encoder.encode(row['False Relation'])
            row['False Relation'] = relation_encoder.encode(row['False Relation'])

        if 'Relation' in row:
            row['Relation Word'] = relation_word_encoder.encode(row['Relation'])
            row['Relation'] = relation_encoder.encode(row['Relation'])

print('Train Sample:')
display(pd.DataFrame(train_dataset[:5]))
print('Development Sample:')
display(pd.DataFrame(dev_dataset[:5]))

# ## Dataset Iterators
#
# Define functions to create iterators over the development and the train dataset for each epoch.

# In[5]:

from functools import partial

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from lib.utils import pad_batch
from lib.samplers import BucketBatchSampler
from lib.samplers import SortedSampler

train_max_batch_size = 128
dev_max_batch_size = 128


# Defines how to combine a batch of rows into a tensor
def collate_fn(batch, train=True):
    """ list of tensors to a batch variable """
    question_batch, _ = pad_batch([row['Question'] for row in batch])

    # PyTorch RNN requires batches to be transposed for speed and integration with CUDA
    to_variable = (lambda b: Variable(torch.stack(b).t_().contiguous(), volatile=not train))

    if train:
        true_relation_word_batch, _ = pad_batch([row['True Relation Word'] for row in batch])
        true_relation_batch, _ = pad_batch([row['True Relation'] for row in batch])
        false_relation_word_batch, _ = pad_batch([row['False Relation Word'] for row in batch])
        false_relation_batch, _ = pad_batch([row['False Relation'] for row in batch])
        return (to_variable(question_batch), to_variable(true_relation_batch),
                to_variable(true_relation_word_batch), to_variable(false_relation_batch),
                to_variable(false_relation_word_batch))
    else:
        relation_word_batch, _ = pad_batch([row['Relation Word'] for row in batch])
        relation_batch, _ = pad_batch([row['Relation'] for row in batch])
        return (to_variable(question_batch), to_variable(relation_batch),
                to_variable(relation_word_batch), batch)


def make_train_iterator():
    # Use bucket sampling to group similar sized text but with noise + random
    sort_key = lambda r: r['Question'].size()[0]
    batch_sampler = BucketBatchSampler(train_dataset, sort_key, train_max_batch_size)
    return DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        pin_memory=is_cuda,
        num_workers=0)


def make_dev_iterator():
    # Group together all examples for metrics and sort questions of similar sizes for speed
    sort_key = lambda r: (r['Question'].size()[0], r['Example ID'])
    return DataLoader(
        dev_dataset,
        batch_size=dev_max_batch_size,
        sampler=SortedSampler(dev_dataset, sort_key, sort_noise=0.0),
        collate_fn=partial(collate_fn, train=False),
        pin_memory=is_cuda,
        num_workers=0)


# Just to make sure everything runs
train_iterator_test = make_train_iterator()
dev_iterator_test = make_dev_iterator()

# # Loss
#
# Instantiate the loss.

# In[6]:

from torch.nn.modules.loss import MarginRankingLoss

# QUESTION: Is there a better margin? or wrose?
criterion = cuda(MarginRankingLoss(margin=0.5))

# # Model
#
# Instantiate the model.

# In[7]:

# In[8]:

import torch
from lib.pretrained_embeddings import FastText

# Load embeddings
unk_init = lambda t: torch.FloatTensor(t).uniform_(-0.1, 0.1)
pretrained_embedding = FastText(language='en', cache='./.pretrained_embeddings_cache')
embedding_weights = torch.Tensor(text_encoder.vocab_size, pretrained_embedding.dim)
for i, token in enumerate(text_encoder.vocab):
    embedding_weights[i] = pretrained_embedding[token]

# In[9]:

from lib.nn import YuModel


def make_model():
    model = YuModel(relation_encoder.vocab_size, relation_word_encoder.vocab_size,
                    text_encoder.vocab_size)
    for param in model.parameters():
        param.data.uniform_(-0.1, 0.1)

    model.text_embedding.weight.data.copy_(embedding_weights)

    cuda(model)
    return model


# ## Gradient Descent Optimizer
#
# Instantiate the gradient descent optimizer.

# In[10]:

from torch.optim import Adam

from lib.optim import Optimizer


# https://github.com/pytorch/pytorch/issues/679
# TODO: Try SGD
def make_optimizer(model):
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Optimizer(Adam(params=params))
    return optimizer


# ## Training Loop
#
# Below here, we do a training loop over a number of epochs.

# In[11]:

from lib.utils import get_total_parameters
from lib.utils import get_log_directory_path

epochs = 30
log_directory = get_log_directory_path('yu_relation_model')
model = make_model()
optimizer = make_optimizer(model)

print('Log Directory: %s' % log_directory)
print('Devevelopment Max Batch Size: %s' % dev_max_batch_size)
print('Train Max Batch Size: %s' % train_max_batch_size)
print('Epochs: %s' % epochs)
print('Total Parameters: %d' % get_total_parameters(model))
print('Model:\n%s' % model)

# In[12]:

from tqdm import tqdm
from collections import defaultdict

from lib.checkpoint import Checkpoint

# Train!
for epoch in range(epochs):
    print('Epoch %d' % epoch)

    # Iterate over the training data
    model.train(mode=True)
    train_iterator = make_train_iterator()
    for (question, true_relation, true_relation_word, false_relation,
         false_relation_word) in tqdm(train_iterator):
        optimizer.zero_grad()
        output_true = model(
            cuda_async(question), cuda_async(true_relation), cuda_async(true_relation_word))
        output_false = model(
            cuda_async(question), cuda_async(false_relation), cuda_async(false_relation_word))
        labels = cuda(Variable(torch.ones(output_true.size()[0])))
        loss = criterion(output_true, output_false, labels)

        # Backward propagation
        loss.backward()
        optimizer.step()

    # Save checkpoint
    Checkpoint.save(
        log_directory, {
            'model': model,
            'optimizer': optimizer,
            'relation_word_encoder': relation_word_encoder,
            'relation_encoder': relation_encoder,
            'text_encoder': text_encoder
        },
        device=device)

    # Evaluate
    model.train(mode=False)
    examples = defaultdict(list)
    dev_iterator = make_dev_iterator()
    total_loss = 0
    for (question, relation, relation_word, batch) in tqdm(dev_iterator):
        output = model(cuda_async(question), cuda_async(relation), cuda_async(relation_word))
        output = output.data.cpu()

        for i, row in enumerate(batch):
            examples[row['Example ID']].append({
                'Score': output[i],
                'Question': row['Question'],
                'True Relation': row['True Relation'],
                'Relation': row['Relation']
            })

    # Print metrics
    correct = 0
    for pool in examples.values():
        max_relation = max(pool, key=lambda p: p['Score'])
        if max_relation['Relation'].tolist() == max_relation['True Relation'].tolist():
            correct += 1
    print('Accuracy: %f [%d of %d]' % (correct / len(examples), correct, len(examples)))
    print()
