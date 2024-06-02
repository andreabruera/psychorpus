import argparse
import logging
import multiprocessing
import os
import re
import spacy
import sys

from tqdm import tqdm

sys.path.append('..')
from readers import cc100_original_reader, paths_loader

def preprocess(ins):

    file_path = ins[0]
    out_folder = ins[1]  
    n = ins[2]

    all_sentences = cc100_original_reader(args, file_path)
    cleaned_lines = list()
    for sentence in tqdm(all_sentences):
        l = ' '.join(sentence['word'])

        ### annotating with spacy
        spacy_l = model(l)
        ner_annotation = [ent.ent_iob_ for ent in spacy_l]
        entity_indices = [w_i for w_i, w in enumerate(ner_annotation) if w=='B']
        removed_indices = [w_i for w_i, w in enumerate(ner_annotation) if w=='I']
        ents = spacy_l.ents
        assert len(entity_indices) == len(ents)

        for t_i, token in enumerate(spacy_l):

            if t_i in entity_indices:
                entity_ind = entity_indices.index(t_i)
                ent = ents[entity_ind].text
                ent_type = ents[entity_ind].label_
                line = [ent, ent, 'ENT', ent_type]
                cleaned_lines.append(line)
            elif t_i not in removed_indices:
                w = token.text
                lemma = token.lemma_
                pos = token.pos_
                line = [w, lemma, 'WORD', pos]
                cleaned_lines.append(line)
        cleaned_lines.append(['<EOS>', '<EOS>', '<EOS>', '<EOS>'])

    ### writing to file
    full_out_path = os.path.join(out_folder,  '{:010}.cc100_tagged'.format(n))
    with open(full_out_path, 'w', encoding='utf-8') as o:
        o.write('Word\tLemma\tWord type\tEntity type or POS\n')
        for l in cleaned_lines:
            for w in l:
                o.write('{}\t'.format(w))
            o.write('\n')
    print([len(cleaned_lines), full_out_path])

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logging.info('Now loading Spacy')

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
                    default='cc100',
                    )
global args
args = parser.parse_args()

global model

if args.language == 'it':
    model = spacy.load('it_core_news_lg')
if args.language == 'en':
    model = spacy.load('en_core_web_lg')
if args.language == 'de':
    model = spacy.load('de_core_news_lg')

out_folder = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'corpora', args.language, 'cc100-{}_tagged'.format(args.language))
os.makedirs(out_folder, exist_ok=True)
paths = [(p, out_folder, p_i) for p_i, p in enumerate(paths_loader(args))]

logging.info('Now annotating CC100!')

if __name__ == '__main__':
    
    with multiprocessing.Pool(processes=int((os.cpu_count()/3)*2)) as p:
        p.map(preprocess, paths)
    p.terminate()
    p.join()
