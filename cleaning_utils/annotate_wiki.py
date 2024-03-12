import argparse
import logging
import multiprocessing
import os
import re
import spacy

from .readers import wiki_reader, wac_reader, bnc_reader, opensubs_reader, paths_loader

def preprocess(folder_path):
    all_sentences, file_paths = wiki_reader(args, folder_path, file_paths=True)
    old_path = ''
    o = open(os.path.join('..', '__pycache__', 'rubbish'), 'w')
    for sentence, path in zip(all_sentences, file_paths):
        cleaned_lines = list()
        l = ' '.join(sentence['word']).replace('[[[', 'ENTSTART').replace(']]]', 'ENTEND')

        ### annotating with spacy
        spacy_l = model(l)
        ner_annotation = [ent.ent_iob_ for ent in spacy_l]
        entity_indices = [w_i for w_i, w in enumerate(ner_annotation) if w=='B']
        removed_indices = [w_i for w_i, w in enumerate(ner_annotation) if w=='I']
        ents = spacy_l.ents
        assert len(entity_indices) == len(ents)

        for t_i, token in enumerate(spacy_l):

            if 'ENTSTART' in token.text and 'ENTEND' in token.text:
                ent = token.text.replace('ENTSTART', '').replace('ENTEND', '').replace('_', ' ')
                line = [ent, ent, 'ENT', 'WIKI_ENT']
                cleaned_lines.append(line)
            elif t_i in entity_indices:
                entity_ind = entity_indices.index(t_i)
                ent = ents[entity_ind].text.replace('ENTSTART', '').replace('ENTEND', '').replace('_', ' ')
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
        out_path = folder_path.replace('art_by_art', 'tagged_art_by_art')
        os.makedirs(out_path, exist_ok=True)
        full_out_path = os.path.join(out_path, path)
        print(full_out_path)
        if full_out_path != old_path:
            o.close()
            o = open(full_out_path, 'w', encoding='utf-8')
            o.write('Word\tLemma\tWord type\tEntity type or POS\n')
            old_path = '{}'.format(full_out_path)
        for l in cleaned_lines:
            for w in l:
                o.write('{}\t'.format(w))
            o.write('\n')

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
                    choices=[
                             'wiki', 
                             'wac', 
                             'bnc', 
                             'opensubs',
                             ],
                    default='wiki',
                    )
global args
args = parser.parse_args()

global model

if args.language == 'it':
    model = spacy.load('it_core_news_lg')
if args.language == 'en':
    model = spacy.load('en_core_web_lg')
if args.language == 'de':
    model = spacy.load('en_core_news_lg')

paths = paths_loader(args)

logging.info('Now annotating Wikipedia!')

if __name__ == '__main__':
    
    with multiprocessing.Pool(processes=int(os.cpu_count()/3)) as p:
        p.map(preprocess, paths)
    p.terminate()
    p.join()
