import argparse
import multiprocessing
import numpy
import os
import pickle
import re

from tqdm import tqdm

from readers import wiki_reader, wac_reader, bnc_reader, opensubs_reader, paths_loader

def multiprocessing_counter(file_path):

    if args.corpus == 'wiki':
        all_sentences = wiki_reader(file_path)
    if args.corpus == 'wac':
        all_sentences = wac_reader(file_path)
    if args.corpus == 'opensubs':
        all_sentences = opensubs_reader(file_path)
    if args.corpus == 'bnc':
        all_sentences = bnc_reader(file_path)

    cased = dict()
    uncased = dict()
    with tqdm() as counter:
        for sentence in all_sentences:
            #print(sentence)
            for w in sentence['word']:
                try:
                    cased[w] += 1
                    uncased[w.lower()] += 1
                except KeyError:
                    cased[w] = 1
                    uncased[w.lower()] = 1
            counter.update(1)
    return (cased, uncased)

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

pkls = os.path.join('pickles', args.language, args.corpus)
os.makedirs(pkls, exist_ok=True)

cas_freqs_file = os.path.join(pkls, '{}_{}_cased_word_freqs.pkl'.format(args.language, args.corpus))
uncas_freqs_file = os.path.join(pkls, '{}_{}_uncased_word_freqs.pkl'.format(args.language, args.corpus))

paths = paths_loader(args)

### Running
with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
   results = pool.map(multiprocessing_counter, paths)
   pool.terminate()
   pool.join()

all_cased = dict()
all_uncased = dict()
### Reorganizing results
print('now reorganizing multiprocessing results...')
for cased, uncased in tqdm(results):
    for k, v in cased.items():
        try:
            all_cased[k] += v
        except KeyError:
            all_cased[k] = v
    for k, v in uncased.items():
        try:
            all_uncased[k] += v
        except KeyError:
            all_uncased[k] = v

with open(cas_freqs_file, 'wb') as o:
    pickle.dump(all_cased, o)
with open(uncas_freqs_file, 'wb') as o:
    pickle.dump(all_uncased, o)
