import argparse
import fasttext
import gensim
import numpy
import os
import pickle
import random
import scipy

import gensim.downloader as api

from scipy import spatial, stats
from tqdm import tqdm

from utils import build_ppmi_vecs, read_ratings, read_men, read_men_test, read_simlex, read_fernandino

ratings = read_ratings(hand=True)

corpus = [
               'opensubs',
               #'tagged_wiki',
               #'bnc',
               #'wac',
               ][0]
min_count = [
                  10, 
                  #100,
                  ][0]
percent_val = [
                    #0.001, 
                    #0.01, 
                    #0.1, 
                    1,
                    ][0]
win_size = [
                 4, 
                 #20,
                 ][0]

print('\n\n{} - min {} - {} percent - window size {}'.format(corpus, min_count, percent_val*100, win_size))
with open(os.path.join(
                       'pickles', 'en', corpus, 
                       'en_{}_uncased_word_freqs.pkl'.format(corpus),
                       ), 'rb') as i:
    freqs = pickle.load(i)
with open(os.path.join(
                       'pickles', 'en', corpus, 
                       'en_{}_uncased_word_pos.pkl'.format(corpus),
                       ), 'rb') as i:
    pos = pickle.load(i)
print('total size of the corpus: {} tokens'.format(sum(freqs.values())))
try:
    with open(os.path.join(
                           'pickles', 'en', corpus, 
                           'en_{}_coocs_uncased_min_{}_win_{}.pkl'.format(corpus, min_count, win_size),
                           #'en_{}_coocs_uncased_min_{}_win_4.pkl'.format(corpus, min_count),
                           ), 'rb') as i:
        coocs = pickle.load(i)
except FileNotFoundError:
    raise RuntimeError('this model is missing!')
with open(os.path.join(
                       'pickles', 'en', corpus, 
                       'en_{}_uncased_vocab_min_{}.pkl'.format(corpus, min_count),
                       ), 'rb') as i:
    vocab = pickle.load(i)

words, data = read_fernandino(vocab, pos)

test_words = set(words[1]).union(set(words[2]))
present=dict()
print('missing words:')
missing = list()
for w in test_words:
    if w not in freqs.keys():
        missing.append(w)
        print('{} - not appearing at all'.format(w))
        continue
    if vocab[w]==0:
        missing.append(w)
        print([m, pos[m]])
        continue
    else:
        present[w] = freqs[w]
print('\n')
print('corpus: {}'.format(corpus))
print('number of words missing: {}'.format(len(missing)))
min_n = min(list(present.values()))
max_n = max(list(present.values()))
lower_ten_ws = [w for w, val in present.items() if val < 10]
lower_fifty_ws = [w for w, val in present.items() if val < 50]
lower_hundred_ws = [w for w, val in present.items() if val < 100]
ten_n = sum([1 for val in present.values() if val >= 10])
fifty_n = sum([1 for val in present.values() if val >= 50])
hundred_n = sum([1 for val in present.values() if val >= 100])
avg_n = numpy.average(list(present.values()))
med_n = numpy.median(list(present.values()))
print('minimum number of mentions: {}'.format(min_n))
print('maximum number of mentions: {}'.format(max_n))
print('number of items above 10 mentions: {}'.format(ten_n))
print('items with less than 10 mentions:')
print(lower_ten_ws)
print('number of items above 50 mentions: {}'.format(fifty_n))
print('items with less than 50 mentions:')
print(lower_fifty_ws)
print('number of items above 100 mentions: {}'.format(hundred_n))
print('items with less than 100 mentions:')
print(lower_hundred_ws)
print('average number of mentions: {}'.format(avg_n))
print('median number of mentions: {}'.format(med_n))

print('total number of ratings words available: {}'.format(len(ratings.keys())))
### removing rare words
pruned_test_words = [w for w in test_words if w not in missing]
#freq_missing_ratings = set(ratings.keys()).difference(set(freqs.keys()))
#voc_missing_ratings = [w for w, dct in ratings.items() if w in freqs.keys() and vocab[w]==0]
#freq_missing_ratings = [w for w, dct in ratings.items() if w not in freqs.keys()]
pruned_ratings = {w : dct for w, dct in ratings.items() if w in freqs.keys() and vocab[w]!=0}
print('number of ratings words actually used: {}'.format(len(pruned_ratings.keys())))
#percent = int(len(pruned_ratings.items())*0.001)
#percent = int(len(pruned_ratings.items())*0.06)
percent = int(len(pruned_ratings.items())*percent_val)
### all words
#percent = int(len(pruned_ratings.items()))
### context words
### things improve when including the words directly
ctx_words = set(pruned_test_words)
#ctx_words = set()
sem_dims = set([var for k,v in pruned_ratings.items() for var in v.keys()])
for dim in sem_dims:
    sorted_ws = sorted([(w, v[dim]) for w, v in pruned_ratings.items()], key=lambda item: item[1])
    ctx_words = ctx_words.union(set([w for w, val in sorted_ws[-percent:]]))
    if dim == 'concreteness':
    #    continue
        ### also adding super abstract words
        ctx_words = ctx_words.union(set([w[0] for w in sorted_ws[:percent]]))
    #    ctx_words = ctx_words.union(set([w for w, val in sorted_ws[-one_percent:]]))
print('considering {} context words'.format(len(ctx_words)))
ctx_words = sorted(ctx_words)
ctx_idxs = [vocab[w] for w in ctx_words]
vecs = {w : numpy.array([coocs[vocab[w]][idx] if idx in coocs[vocab[w]].keys() else 0 for idx in ctx_idxs]) for w in pruned_test_words}
### pmi
### building the PPMI matrix
### things are better when including in the rows the words from MEN...
trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, ctx_words, ctx_words)
### re-building the matrix
mtrx = numpy.array([trans_pmi_vecs[w] for w in ctx_words])
assert mtrx.shape == (len(ctx_words), len(ctx_words))
with open(os.path.join(
                       'pickles', 'en', corpus, 
                       'en_{}_coocs_uncased_min_{}_win_{}_pmi_rows_cols.pkl'.format(corpus, min_count, win_size),
                       ), 'wb') as i:
    pickle.dump(ctx_words, i)
with open(os.path.join(
                       'pickles', 'en', corpus, 
                       'en_{}_coocs_uncased_min_{}_win_{}_pmi.pkl'.format(corpus, min_count, win_size),
                       ), 'wb') as i:
    pickle.dump(mtrx, i)
rng = numpy.random.default_rng()
### shuffled
curr_mtrx = numpy.copy(mtrx)
rng.shuffle(curr_mtrx, axis=1)
with open(os.path.join(
                       'pickles', 'en', corpus, 
                       'en_{}_coocs_uncased_min_{}_win_{}_rand.pkl'.format(corpus, min_count, win_size),
                       ), 'wb') as i:
    pickle.dump(curr_mtrx, i)
### powers
### power + shuffle
for power in [
              0.05, 0.1,
              #0.25, 0.5, 0.75,
              ]:
    with open(os.path.join(
                           'pickles', 'en', corpus, 
                           'en_{}_coocs_uncased_min_{}_win_{}_rand_pow_{}.pkl'.format(corpus, min_count, win_size, power),
                           ), 'wb') as i:
        pickle.dump(numpy.power(mtrx, power), i)
    curr_mtrx = numpy.copy(mtrx)
### power only
for power in [0.05, 0.1,
               #0.25, 0.5, 0.75,
               ]:
    curr_mtrx = numpy.copy(mtrx)
    with open(os.path.join(
                           'pickles', 'en', corpus, 
                           'en_{}_coocs_uncased_min_{}_win_{}_pow_{}.pkl'.format(corpus, min_count, win_size, power),
                           ), 'wb') as i:
        pickle.dump(numpy.power(mtrx, power), i)
    curr_mtrx = numpy.copy(mtrx)
