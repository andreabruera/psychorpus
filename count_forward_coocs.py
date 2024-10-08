import argparse
import multiprocessing
import numpy
import os
import pickle
import re

from tqdm import tqdm

from readers import cc100_original_reader, tagged_wiki_reader, wiki_reader, wac_reader, bnc_reader, opensubs_reader, paths_loader

def coocs_counter(keyed_sentence, coocs):
    for start_i, start in enumerate(keyed_sentence):
        if start == 0:
            continue
        #sent_slice = keyed_sentence[min(0, start_i-half_win):start_i] + keyed_sentence[start_i+1:start_i+half_win]
        sent_slice = keyed_sentence[start_i+1:start_i+win]
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
    if args.corpus == 'tagged_wiki':
        all_sentences = tagged_wiki_reader(args, file_path)
    if args.corpus == 'wiki':
        all_sentences = wiki_reader(args, file_path)
    if args.corpus == 'wac':
        all_sentences = wac_reader(args, file_path)
    if args.corpus == 'opensubs':
        all_sentences = opensubs_reader(args, file_path)
    if args.corpus == 'bnc':
        all_sentences = bnc_reader(args, file_path)
    if args.corpus == 'cc100':
        all_sentences = cc100_original_reader(args, file_path)

    for sentence in tqdm(all_sentences):
        #print(sentence)
        keyed_sentence = list()
        for w in sentence['word']:
            if args.case == 'uncased':
                w = w.lower()
            try:
                idx = vocab[w]
            except KeyError:
                idx = 0
            keyed_sentence.append(idx)

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
                             20,
                             50, 
                             100,
                             500,
                             1000,
                             ],
                    required=True,
                    type=int,
                    )
parser.add_argument(
                    '--window_size', 
                    choices=[
                             4,
                             5,
                             10, 
                             15, 
                             20,
                             30,
                             ],
                    required=True,
                    type=int,
                    )
parser.add_argument(
                    '--corpus', 
                    choices=[
                             'tagged_wiki', 
                             'wiki', 
                             'wac', 
                             'bnc', 
                             'opensubs',
                             'cc100',
                             ],
                    required=True,
                    )
parser.add_argument(
                    '--no_entities', 
                    action='store_true',
        )
global args
args = parser.parse_args()

if args.no_entities:
    selected_pos = ['PROPN', 'VERB', 'NOUN', 'ADV', 'ADJ', ]
else:
    selected_pos = ['PROPN', 'VERB', 'NOUN', 'ADV', 'ADJ', 'ENT',]

global win
win = int(args.window_size)

pkls = os.path.join(
                    #'pickles', 
                    '/',
                    'data',
                    'tu_bruera',
                    'counts',
                    args.language, 
                    args.corpus, 
                    )
assert os.path.exists(pkls)

freqs_file = os.path.join(pkls, '{}_{}_{}_word_freqs.pkl'.format(args.language, args.corpus, args.case))
with open(freqs_file, 'rb') as i:
    print('loading freqs')
    freqs = pickle.load(i)
    print('loaded!')
if args.corpus in ['wiki', 'cc100']:
    ### wiki does not come with pos tagging...
    pos_file = os.path.join(pkls.replace(args.corpus, 'tagged_wiki'), '{}_tagged_wiki_{}_word_pos.pkl'.format(args.language, args.case))
else:
    pos_file = os.path.join(pkls, '{}_{}_{}_word_pos.pkl'.format(args.language, args.corpus, args.case))
with open(pos_file, 'rb') as i:
    print('loading pos')
    pos = pickle.load(i)
    print('loaded!')

global vocab
if args.no_entities:
    vocab_file = os.path.join(pkls, '{}_{}_{}_vocab_min_{}_no-entities.pkl'.format(args.language, args.corpus, args.case, args.min_mentions))
else:
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
            try:
                ### only if dominant
                #dom_pos = sorted(list(pos[k].items()), key=lambda item : item[1], reverse=True)[0][0]
                #if dom_pos in selected_pos:
                #    vocab[k] = counter
                #    counter += 1
                ### soft threshold at 0.3
                fith = sum([val for val in pos[k].values()])*0.2
                dom_pos = [val for val, count in pos[k].items() if count>fifth]
                marker = False
                for sel in selected_pos:
                    if sel in dom_pos:
                        marker = True
                if marker:
                    vocab[k] = counter
                    counter += 1
                else:
                    vocab[k] = 0
            except KeyError:
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
if args.no_entities:
    coocs_file = os.path.join(pkls, '{}_{}_forward-coocs_{}_min_{}_win_{}_no-entities.pkl'.format(args.language, args.corpus, args.case, args.min_mentions, args.window_size))
else:
    coocs_file = os.path.join(pkls, '{}_{}_forward-coocs_{}_min_{}_win_{}.pkl'.format(args.language, args.corpus, args.case, args.min_mentions, args.window_size))

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
