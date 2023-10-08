import argparse
import multiprocessing
import os
import pickle
import re
import spacy

from tqdm import tqdm

def read_wac(file_path):
    with open(file_path) as i:
        sentence = list()
        for l in i:
            line = l.strip().split('\t')
            #if line[0][:5] in ['<sent', '<erro', '<sour', '<year']: 
            #    continue
            #elif line[0][:3] == '<s>':
            #    continue
            #elif line[0][:6] == '</sent':
            #    continue
            if line[0][:4] == '</s>':
                sentence = ' '.join(sentence)
                yield sentence
                sentence = list()
            elif line[0][0] == '<':
                continue

            if len(line) < 2:
                continue
            else:
                if '$' in line[1]:
                    continue
                else:
                    sentence.append(line[0])
        yield ' '.join(sentence)

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
    lemma_counter = dict()
    with tqdm() as counter:
        if args.dataset == 'wac':
            sents = read_wac(file_path)
        elif args.dataset == 'opensubs':
            sents = read_opensubs(file_path)
        for sentence in sents:
            spacy_sentence = [(w.text, w.pos_) for w in spacy_model(sentence)]
            #print(spacy_sentence)
            for word, pos in spacy_sentence:
                ### words
                try:
                    word_counter[(word, pos)] += 1
                except KeyError:
                    word_counter[(word, pos)] = 1
                counter.update(1)
    return word_counter

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--corpora_folder', 
                    required=True,
                    help='path to the folder containing '
                    'the corpora'
                    )
parser.add_argument('--language', choices=['it', 'en', 'de'], required=True)
parser.add_argument('--dataset', choices=['wac', 'opensubs'], required=True)
args = parser.parse_args()

if args.dataset == 'wac':
    if args.language == 'en':
        dataset_folder = 'PukWaC_smaller_files'
    if args.language == 'de':
        dataset_folder = 'sdewac-v3-tagged_smaller_files'
    elif args.language == 'it':
        dataset_folder = 'itwac_smaller_files'
    min_files = 400
elif args.dataset == 'opensubs':
    dataset_folder = 'opensubs_ready'
    min_files = 4000

corpus_path = os.path.join(args.corpora_folder, args.language, dataset_folder)
try:
    assert os.path.exists(corpus_path)
except AssertionError:
    raise RuntimeError('The path provided for Wac does not exist!')
paths = [os.path.join(corpus_path, f) for f in os.listdir(corpus_path)]
try:
    assert len(paths) > min_files
except AssertionError:
    raise RuntimeError('split dataset is composed by more than {} files, but '
                       'the provided folder contains less'.format(min_files)
                       )

pkls = os.path.join('pickles', args.language)
os.makedirs(pkls, exist_ok=True)
global spacy_model
spacy_model = spacy.load('en_core_web_lg')

word_freqs_file = os.path.join(pkls, '{}_{}_word_pos.pkl'.format(args.language, args.dataset))
if os.path.exists(word_freqs_file):
    with open(word_freqs_file, 'rb') as i:
        print('loading word pos')
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
    for freq_dict in results:
        ### words
        for k_pos, v in freq_dict.items():
            try:
                word_freqs[k_pos] += v
            except KeyError:
                word_freqs[k_pos] = v
    ### final passage
    words = set([v[0] for v in word_freqs.keys()])
    final_dict = {w : dict() for w in words}
    for k, v in word_freqs.items():
        final_dict[k[0]][k[1]] = v

    with open(word_freqs_file, 'wb') as o:
        pickle.dump(final_dict, o)
