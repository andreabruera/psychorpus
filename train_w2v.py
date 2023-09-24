import argparse
import gensim
import logging
import os
import re
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class corpus_reader:
    def __iter__(self):
        for c in corpora:
            for f in os.listdir(c):
                if 'web' in f:
                    continue
                if 'opensubs_ready' in c:
                    ### grouping by chunks of 512 tokens
                    sentence = list()
                    with open(os.path.join(c, f)) as i:
                        for l in i:
                            line = re.sub(r'-', r'', l)
                            line = re.sub('\W', ' ', line)
                            line = re.sub('\s+', r' ', line)
                            line = line.split()
                            sentence.extend(line)
                            if len(sentence) >= 512:
                                yield sentence
                                sentence = list()
                        if len(sentence) > 1:
                            yield(sentence)
                else:
                    with open(os.path.join(c, f), encoding=encoding) as i:
                        marker = True
                        sentence = list()
                        for l in i:
                            line = l.strip().split('\t')
                            #print(line)
                            '''
                            if line[0][:5] in ['<sent', '<erro', '<sour', '<year']: 
                                continue
                            elif line[0][:3] == '<s>':
                                continue
                            elif line[0][:6] == '</sent':
                                continue
                            '''
                            if line[0][:4] == '</s>':
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
global corpora
global encoding

parser = argparse.ArgumentParser()
parser.add_argument('--language', choices=['it', 'en', 'de'], required=True)
args = parser.parse_args()

if args.language == 'en':
    wac_folder = 'PukWaC_smaller_files'
    encoding = 'utf-8'
if args.language == 'de':
    wac_folder = 'sdewac-v3-tagged_smaller_files'
    encoding = 'utf-8'
elif args.language == 'it':
    wac_folder = 'itwac'
    encoding = 'latin-1'

corpora_folder = [
           'opensubs_ready', 
           wac_folder
           ]
corpora = [os.path.join(args.language, c) for c in corpora_folder]

model = Word2Vec(
                 sentences=corpus_reader(), 
                 size=300, 
                 window=6, 
                 min_count=100, 
                 workers=int(os.cpu_count()/2),
                 negative=10,
                 sg=0,
                 sample=1e-5,
                 )
model.save("word2vec_{}_opensubs+wac_param-mandera2017_min-count-100.model".format(args.language))
