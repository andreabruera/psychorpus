import argparse
import multiprocessing
import os
import pickle
import random

from tqdm import tqdm

def read_wac(file_path):
    with open(file_path) as i:
        sentence = {
                    'word' : list(), 
                    'lemma' : list(),
                    }
        for l in i:
            line = l.strip().split('\t')
            #if line[0][:5] in ['<sent', '<erro', '<sour', '<year']: 
            #    continue
            #elif line[0][:3] == '<s>':
            #    continue
            #elif line[0][:6] == '</sent':
            #    continue
            if line[0][:4] == '</s>':
                yield sentence
                sentence = {
                    'word' : list(), 
                    'lemma' : list(),
                    }
            elif line[0][0] == '<':
                continue

            if len(line) < 2:
                continue
            else:
                if '$' in line[1]:
                    continue
                else:
                    sentence['word'].append('{}_{}'.format(line[0], line[1]))
                    sentence['lemma'].append('{}_{}'.format(line[2], line[1]))

def collector(file_path):
    word_collector = {w : list() for w in words.keys()}
    lemma_collector = {w : list() for w in words.keys()}
    with tqdm() as counter:
        for sentence in read_wac(file_path):
            for word, lemma, idx in zip(sentence['word'], sentence['lemma'], range(len(sentence['word']))):
                clean_w = word.split('_')
                clean_lem = lemma.split('_')
                for relevant_w, pos_markers in words.items():
                    ### words
                    if clean_w[0].lower() == relevant_w and clean_w[1] in pos_markers:
                        ready_sent = [w.split('_')[0] if w_i!=idx else '[SEP] {} [SEP]'.format(w.split('_')[0]) for w_i, w in enumerate(sentence['word'])]
                        word_collector[relevant_w].append(' '.join(ready_sent))
                        counter.update(1)
                    ### lemmas
                    if clean_lem[0] == relevant_w and clean_lem[1] in pos_markers:
                        ready_sent = [w.split('_')[0] if w_i!=idx else '[SEP] {} [SEP]'.format(w.split('_')[0]) for w_i, w in enumerate(sentence['word'])]
                        lemma_collector[relevant_w].append(' '.join(ready_sent))
                        counter.update(1)
    return word_collector, lemma_collector

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--wac_path', 
                    required=True,
                    help='path to the folder containing '
                    'the files for the wac dataset'
                    )
parser.add_argument('--language', choices=['it', 'en', 'de'], required=True)
parser.add_argument(
                    '--marker',
                    choices=[
                             'unaware_semantics',
                             ],
                    required=True,
                    help='marker for the words dataset'
                    )
args = parser.parse_args()

if args.language == 'en':
    wac_folder = 'PukWaC_smaller_files'
if args.language == 'de':
    wac_folder = 'sdewac-v3-tagged_smaller_files'
elif args.language == 'it':
    wac_folder = 'itwac_smaller_files'

### reading files
global words
words = dict()
with open(os.path.join('data', 'experiments_word_lists', '{}.txt'.format(args.marker))) as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        words[line[0]] = line[1:]

try:
    assert os.path.exists(args.wac_path)
except AssertionError:
    raise RuntimeError('The path provided for Wac does not exist!')
paths = [os.path.join(args.wac_path, f) for f in os.listdir(args.wac_path)]
try:
    assert len(paths) > 400
except AssertionError:
    raise RuntimeError('(split) Wac is composed by more than 400 files, but '
                       'the provided folder contains less'
                       )

pkls = os.path.join('pickles', args.language)
os.makedirs(pkls, exist_ok=True)

word_sents_file = os.path.join(pkls, '{}_{}_wac_word_sents.pkl'.format(args.marker, args.language))
lemma_sents_file = os.path.join(pkls, '{}_{}_wac_lemma_sents.pkl'.format(args.marker, args.language))
if os.path.exists(word_sents_file):
    with open(word_sents_file, 'rb') as i:
        print('loading word sents')
        word_sents = pickle.load(i)
        print('loaded!')
    with open(lemma_sents_file, 'rb') as i:
        print('loading lemma sents')
        lemma_sents = pickle.load(i)
        print('loaded!')
else:

    ### Running
    with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
       results = pool.map(collector, paths)
       pool.terminate()
       pool.join()

    ### Reorganizing results
    word_sents = {w : list() for w in words.keys()}
    lemma_sents = {w : list() for w in words.keys()}
    for sent_dict in results:
        ### words
        for k, v in sent_dict[0].items():
            word_sents[k].extend(v)
        ### lemmas
        for k, v in sent_dict[1].items():
            lemma_sents[k].extend(v)

    with open(word_sents_file, 'wb') as o:
        pickle.dump(word_sents, o)
    with open(lemma_sents_file, 'wb') as o:
        pickle.dump(lemma_sents, o)

marker_folder = os.path.join('sentences', args.marker)
os.makedirs(marker_folder, exist_ok=True)
### writing to file
for w, sentences in lemma_sents.items():
    random_sents = random.sample(sentences, k=min(1000, len(sentences)))
    with open(os.path.join(marker_folder, '{}_itwac.tsv'.format(w, args.marker)), 'w') as o:
        o.write('corpus\tsentence\n')
        for s in random_sents:
            o.write('itwac\t{}\n'.format(s.replace('\t', ' ')))
