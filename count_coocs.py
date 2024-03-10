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
        if start == 0:
            continue
        sent_slice = keyed_sentence[min(0, start_i-half_win):start_i] + keyed_sentence[start_i+1:start_i+half_win]
        #print(sent_slice)
        for other in sent_slice:
            if other == 0:
                continue
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
        all_sentences = wiki_reader(args, file_path)
    if args.corpus == 'wac':
        all_sentences = wac_reader(args, file_path)
    if args.corpus == 'opensubs':
        all_sentences = opensubs_reader(args, file_path)
    if args.corpus == 'bnc':
        all_sentences = bnc_reader(args, file_path)

    for sentence in tqdm(all_sentences):
        #print(sentence)
        if args.case == 'cased':
            keyed_sentence = [vocab[w] for w in sentence['word']]
        else:
            keyed_sentence = [vocab[w.lower()] for w in sentence['word']]
        coocs = coocs_counter(keyed_sentence, coocs)
    ### remove coocs for 0, which is the empty case
    #try:
    #    del coocs[0]
    #except KeyError:
    #    pass

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
pos_file = os.path.join(pkls, '{}_{}_{}_word_pos.pkl'.format(args.language, args.corpus, args.case))
with open(pos_file, 'rb') as i:
    print('loading pos')
    pos = pickle.load(i)
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
        ### setting the minimum frequency
        if v > args.min_mentions:
            ### checking pos
            dom_pos = sorted(list(pos[k].items()), key=lambda item : item[1], reverse=True)[0][0]
            if dom_pos in ['VERB', 'NOUN', 'ADV', 'ADJ']:
                vocab[k] = counter
                counter += 1
            else:
                vocab[k] = 0
        else:
            vocab[k] = 0
print('number of words in the vocabulary: {}'.format(max(vocab.values())))

with open(vocab_file, 'wb') as o:
    pickle.dump(vocab, o)

paths = paths_loader(args)

print('now preparing the coocs dictionary...')
ids = set(vocab.values())
### setting min counts to 1 for the log transformations
general_coocs = {i_one : dict() for i_one in ids}
final_coocs = general_coocs.copy()
print('ready!')
coocs_file = os.path.join(pkls, '{}_{}_coocs_{}_min_{}_win_{}.pkl'.format(args.language, args.corpus, args.case, args.min_mentions, args.window_size))

for file_path in tqdm(paths):
    general_coocs = multiprocessing_counter([file_path, general_coocs])

'''
### Running
with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
   results = pool.map(multiprocessing_counter, [[file_path, general_coocs] for file_path in paths])
   pool.terminate()
   pool.join()

### Reorganizing results
print('now collecting results from multiprocessing...')
original_len = len(results)
for _ in tqdm(range(original_len)):
    for k, v in results[0].items():
        for k_two, v_two in v.items():
            try:
                final_coocs[k][k_two] += v_two
            except KeyError:
                final_coocs[k][k_two] = v_two
    del results[0]
    assert len(results) == original_len-(_+1)
'''

with open(coocs_file, 'wb') as o:
    pickle.dump(general_coocs, o)
