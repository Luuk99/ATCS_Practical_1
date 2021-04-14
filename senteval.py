# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging

# import torch
import torch

# import own files
from main import *

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'
PATH_TO_VEC = './SentEval/pretrained/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# added global variables for the model and embedding dimension
MODEL = None
EMBED_DIM = 300

# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = EMBED_DIM
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(torch.tensor(params.word_vec[word]))
        if not sentvec:
            vec = torch.zeros((1, 300))
            sentvec.append(vec)
        sentvec = torch.stack(sentvec, dim=0)
        embeddings.append(sentvec)

    # pad into tensor
    sentence_lengths = torch.tensor([x.shape[0] for x in embeddings])
    embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, padding_value=0.0, batch_first=True)

    # pass through the model
    embeddings = MODEL.encoder(embeddings.float(), sentence_lengths)

    # cast back to numpy
    embeddings = embeddings.cpu().detach().numpy()

    # return the embeddings
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# For command line activation
if __name__ == "__main__":
    # added parser for selecting the model
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', default='AWE', type=str,
                        help='What model to use. Default is AWE',
                        choices=['AWE', 'UniLSTM', 'BiLSTM', 'BiLSTMMax'])
    args = parser.parse_args()

    # set the global variables
    if (args.model == 'AWE'):
        MODEL = FullModel.load_from_checkpoint('pl_logs/lightning_logs/awe/checkpoints/epoch=10.ckpt')
        EMBED_DIM = 300
    elif (args.model == 'UniLSTM'):
        MODEL = FullModel.load_from_checkpoint('pl_logs/lightning_logs/unilstm/checkpoints/epoch=11.ckpt')
        EMBED_DIM = 2048
    elif (args.model == 'BiLSTM'):
        MODEL = FullModel.load_from_checkpoint('pl_logs/lightning_logs/bilstm/checkpoints/epoch=13.ckpt')
        EMBED_DIM = 2*2048
    else:
        MODEL = FullModel.load_from_checkpoint('pl_logs/lightning_logs/bilstmmax/checkpoints/epoch=7.ckpt')
        EMBED_DIM = 2*2048

    # run the senteval
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC',
                      'SICKRelatedness', 'SICKEntailment', 'STS14']
    results = se.eval(transfer_tasks)

    # save the results
    torch.save(results, args.model + "SentEvalResults.pt")

    # print the results
    print(results)
