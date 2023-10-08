import argparse
import multiprocessing
import os
import pickle
import re
import spacy

from tqdm import tqdm

def read_opensubs(file_path):
    with open(file_path) as i:
        sentence = list()
        ### grouping by chunks of 512 tokens
        with open(file_path) as i:
            for l in i:
                line = re.sub(r'-', r'', l)
                line = re.sub('\W', ' ', line)
                line = re.sub('\s+', r' ', line)
                line = line.split()
                sentence.extend(line)
                if len(sentence) >= 512:
                    yield ' '.join(sentence)
                    sentence = list()
            if len(sentence) > 1:
                yield(' '.join(sentence))

def counter(file_path):
    word_counter = dict()
    with tqdm() as counter:
        for sentence in read_opensubs(file_path):
            spacy_sentence = [(w.text, w.pos_) for w in spacy_model(sentence)]
            print(spacy_sentence)
            for word, pos in spacy_sentence:
                ### words
                try:
                    word_counter[word][pos] += 1
                except KeyError:
                    word_counter[word][pos] = 1
                counter.update(1)
    return word_counter

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--opensubs_path', 
                    required=True,
                    help='path to the folder containing '
                    'the files for the pUkWac dataset'
                    )
parser.add_argument('--language', choices=['it', 'en', 'de'], required=True)
args = parser.parse_args()

try:
    assert os.path.exists(args.opensubs_path)
except AssertionError:
    raise RuntimeError('The path provided for opensubs does not exist!')
paths = [os.path.join(args.opensubs_path, f) for f in os.listdir(args.opensubs_path)]
try:
    assert len(paths) > 4000
except AssertionError:
    raise RuntimeError('opensubs is composed by more than 4000 files, but '
                       'the provided folder contains more/less'
                       )
global spacy_model
spacy_model = spacy.load('en_core_web_lg')

pkls = os.path.join('pickles', args.language)
os.makedirs(pkls, exist_ok=True)

word_pos_file = os.path.join(pkls, '{}_opensubs_word_pos.pkl'.format(args.language))
if os.path.exists(word_pos_file):
    with open(word_freqs_file, 'rb') as i:
        print('loading word pos')
        word_pos = pickle.load(i)
        print('loaded!')
else:

    ### Running
    with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
       results = pool.map(counter, paths)
       pool.terminate()
       pool.join()

    ### Reorganizing results
    word_pos = dict()
    for pos_dict in results:
        ### words
        for k, pos_v in pos_dict.items():
            for pos, v in pos_v.items():
                try:
                    word_pos[k][pos] += v
                except KeyError:
                    word_pos[k][pos] = v

    with open(word_pos_file, 'wb') as o:
        pickle.dump(word_pos, o)
