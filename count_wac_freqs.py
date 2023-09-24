import argparse
import multiprocessing
import os
import pickle

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
                    sentence['word'].append(line[0])
                    sentence['lemma'].append(line[2])

def counter(file_path):
    word_counter = dict()
    lemma_counter = dict()
    with tqdm() as counter:
        for sentence in read_wac(file_path):
            for word, lemma in zip(sentence['word'], sentence['lemma']):
                ### words
                try:
                    word_counter[word] += 1
                except KeyError:
                    word_counter[word] = 1
                ### lemmas
                try:
                    lemma_counter[lemma] += 1
                except KeyError:
                    lemma_counter[lemma] = 1
                counter.update(1)
    return word_counter, lemma_counter

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--wac_path', 
                    required=True,
                    help='path to the folder containing '
                    'the files for the wac dataset'
                    )
parser.add_argument('--language', choices=['it', 'en', 'de'], required=True)
args = parser.parse_args()

if args.language == 'en':
    wac_folder = 'PukWaC_smaller_files'
if args.language == 'de':
    wac_folder = 'sdewac-v3-tagged_smaller_files'
elif args.language == 'it':
    wac_folder = 'itwac_smaller_files'

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

word_freqs_file = os.path.join(pkls, '{}_wac_word_freqs.pkl'.format(args.language))
lemma_freqs_file = os.path.join(pkls, '{}_wac_lemma_freqs.pkl'.format(args.language))
if os.path.exists(word_freqs_file):
    with open(word_freqs_file, 'rb') as i:
        print('loading word freqs')
        word_freqs = pickle.load(i)
        print('loaded!')
    with open(lemma_freqs_file, 'rb') as i:
        print('loading lemma freqs')
        lemma_freqs = pickle.load(i)
        print('loaded!')
else:

    ### Running
    with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
       results = pool.map(counter, paths)
       pool.terminate()
       pool.join()

    ### Reorganizing results
    word_freqs = dict()
    lemma_freqs = dict()
    for freq_dict in results:
        ### words
        for k, v in freq_dict[0].items():
            try:
                word_freqs[k] += v
            except KeyError:
                word_freqs[k] = v
        ### lemmas
        for k, v in freq_dict[1].items():
            try:
                lemma_freqs[k] += v
            except KeyError:
                lemma_freqs[k] = v

    with open(word_freqs_file, 'wb') as o:
        pickle.dump(word_freqs, o)
    with open(lemma_freqs_file, 'wb') as o:
        pickle.dump(lemma_freqs, o)
