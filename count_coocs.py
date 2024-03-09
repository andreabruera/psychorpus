import argparse
import multiprocessing
import numpy
import os
import pickle
import re

from tqdm import tqdm

from readers import wiki_reader, wac_reader, bnc_reader, opensubs_reader, paths_loader

def coocs_counter(keyed_sentence, coocs):
    for start_i, start in enumerate(keyed_sentence):
        sent_slice = keyed_sentence[min(0, start_i-half_win):start_i] + keyed_sentence[start_i+1:start_i+half_win]
        #print(sent_slice)
        for other in sent_slice:
            try:
                coocs[start][other] += 1
            except KeyError:
                coocs[start][other] = 1
            ### debugging
            #if (start, other) != (0, 0):
            #    print(coocs[start][other])
    return coocs


def multiprocessing_counter(all_args):

    file_path = all_args[0]
    coocs = all_args[1]
    if args.corpus == 'wiki':
        all_sentences = wiki_reader(file_path)
    if args.corpus == 'wac':
        all_sentences = wac_reader(file_path)
    if args.corpus == 'opensubs':
        all_sentences = opensubs_reader(file_path)
    if args.corpus == 'bnc':
        all_sentences = bnc_reader(file_path)

    with tqdm() as counter:
        for sentence in all_sentences:
            #print(sentence)
            if args.case == 'cased':
                keyed_sentence = [vocab[w] for w in sentence['word']]
            else:
                keyed_sentence = [vocab[w.lower()] for w in sentence['word']]
            coocs = coocs_counter(keyed_sentence, coocs)
            counter.update(1)
    return coocs

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--language', 
                    choices=[
                             'it', 
                             'en', 
                             'de',
                             ],
                    required=True,
                    )
parser.add_argument(
                    '--case', 
                    choices=[
                             'cased', 
                             'uncased', 
                             ],
                    default='uncased'
                    )
parser.add_argument(
                    '--min_mentions', 
                    choices=[
                             5,
                             10, 
                             50, 
                             100,
                             ],
                    required=True,
                    type=int,
                    )
parser.add_argument(
                    '--window_size', 
                    choices=[
                             5,
                             10, 
                             15, 
                             30,
                             ],
                    required=True,
                    type=int,
                    )
parser.add_argument(
                    '--corpus', 
                    choices=[
                             'wiki', 
                             'wac', 
                             'bnc', 
                             'opensubs',
                             ],
                    required=True,
                    )
global args
args = parser.parse_args()

global half_win
half_win = int(args.window_size/2)

pkls = os.path.join('pickles', args.language, args.corpus)
assert os.path.exists(pkls)

freqs_file = os.path.join(pkls, '{}_{}_{}_word_freqs.pkl'.format(args.language, args.corpus, args.case))
with open(freqs_file, 'rb') as i:
    print('loading freqs')
    freqs = pickle.load(i)
    print('loaded!')

global vocab
vocab_file = os.path.join(pkls, '{}_{}_{}_vocab_min_{}.pkl'.format(args.language, args.corpus, args.case, args.min_mentions))
if os.path.exists(vocab_file):
    with open(vocab_file, 'rb') as i:
        print('loading the vocab')
        vocab = pickle.load(i)
        print('loaded!')
else:
    vocab = dict()
    counter = 1
    print('creating the vocab...')
    for k, v in tqdm(freqs.items()):
        ### setting the minimum frequency to 10
        if v > args.min_mentions:
            vocab[k] = counter
            counter += 1
        else:
            vocab[k] = 0
print('number of words in the vocabulary: {}'.format(max(vocab.values())))

with open(vocab_file, 'wb') as o:
    pickle.dump(vocab, o)

paths = paths_loader(args)

print('now preparing the coocs dictionary...')
ids = set(vocab.values())
### setting min counts to 1 for the log transformations
coocs = {i_one : dict() for i_one in ids}
final_coocs = coocs.copy()
print('ready!')
coocs_file = os.path.join(pkls, '{}_{}_coocs_{}_min_{}_win_{}.pkl'.format(args.language, args.corpus, args.case, args.min_mentions, args.window_size))

### Running
with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
   results = pool.map(multiprocessing_counter, [[file_path, coocs] for file_path in paths])
   pool.terminate()
   pool.join()

### Reorganizing results
print('now collecting results from multiprocessing...')
for coocs_dict in tqdm(results):
    for k, v in coocs_dict.items():
        for k_two, v_two in v.items():
            try:
                final_coocs[k][k_two] += v_two
            except KeyError:
                final_coocs[k][k_two] = v_two

with open(coocs_file, 'wb') as o:
    pickle.dump(final_coocs, o)
