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
        all_sentences = wiki_reader(args, file_path, pos=True)
    if args.corpus == 'wac':
        all_sentences = wac_reader(args, file_path, pos=True)
    if args.corpus == 'opensubs':
        all_sentences = opensubs_reader(args, file_path, pos=True)
    if args.corpus == 'bnc':
        all_sentences = bnc_reader(args, file_path, pos=True)

    cased = dict()
    with tqdm() as counter:
        for sentence in all_sentences:
            #print(sentence)
            for w, pos in zip(sentence['word'], sentence['pos']):
                try:
                    cased[(w, pos)] += 1
                except KeyError:
                    cased[(w, pos)] = 1
            counter.update(1)
    return cased

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

if args.corpus == 'wiki':
    raise RuntimeError('no pos annotation for this corpus...')

pkls = os.path.join('pickles', args.language, args.corpus)
os.makedirs(pkls, exist_ok=True)

cas_pos_file = os.path.join(pkls, '{}_{}_cased_word_pos.pkl'.format(args.language, args.corpus))
uncas_pos_file = os.path.join(pkls, '{}_{}_uncased_word_pos.pkl'.format(args.language, args.corpus))

paths = paths_loader(args, pos=True)

### Running
with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
   results = pool.map(multiprocessing_counter, paths)
   pool.terminate()
   pool.join()

all_cased = dict()
all_uncased = dict()
### Reorganizing results
print('now reorganizing multiprocessing results...')
for cased in tqdm(results):
    for k, v in cased.items():
        w = k[0]
        pos = k[1]
        ### upper
        try:
            all_cased[w][pos] += v
        except KeyError:
            try:
                all_cased[w][pos] = v
            except KeyError:
                all_cased[w] = dict()
                all_cased[w][pos] = v
        ### lower
        try:
            all_uncased[w.lower()][pos] += v
        except KeyError:
            try:
                all_uncased[w.lower()][pos] = v
            except KeyError:
                all_uncased[w.lower()] = dict()
                all_uncased[w.lower()][pos] = v

with open(cas_pos_file, 'wb') as o:
    pickle.dump(all_cased, o)
with open(uncas_pos_file, 'wb') as o:
    pickle.dump(all_uncased, o)
