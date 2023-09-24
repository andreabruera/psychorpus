import logging
import multiprocessing
import os
import re
import spacy

def preprocess(file_path):

    with open(file_path, encoding='utf-8') as i:
        original_lines = [l.strip() for l in i.readlines()]
    ### splitting lines into pages
    pages = dict()
    for l in original_lines:
        ### removing double spaces, tabs
        l = re.sub('\s+', ' ', l)
        ### checking if it's the title
        if l[:4] == '<doc':
            title = re.findall('(?<=title=").+(?=">)', l)
            assert len(title) == 1
            title = title[0]
            pages[title] = list()
        ### checking if it's the end 
        elif l[:5] == '</doc':
            continue
        ### checking if it's too short
        elif len(l) < 3:
            continue
        ### otherwise, keeping the line
        else:
            pages[title].append(l)

    for title, lines in pages.items():
        ### not considering pages that are too short
        if len(lines) < 3:
            continue
        ### output preparation
        re_ed_title = re.sub('\W', r'_', title)
        out_folder = os.path.join(output_folder, re_ed_title[:3])
        os.makedirs(out_folder, exist_ok=True)
        out_file = os.path.join(out_folder, '{}.wiki_page'.format(re_ed_title))
        ### annotating with spacy
        cleaned_lines = list()
        for l in lines:
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
            cleaned_lines.append(['BREAK', 'BREAK', 'BREAK', 'BREAK'])

        ### writing to file
        with open(out_file, 'w', encoding='utf-8') as o:
            o.write('Word\tLemma\tWord type\tEntity type or POS\n')
            for l in cleaned_lines:
                for w in l:
                    o.write('{}\t'.format(w))
                o.write('\n')

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logging.info('Now loading Spacy')

global input_folder
global output_folder
global model

model = spacy.load('it_core_news_lg')

output_folder = os.path.join('it', 'wikipedia_09-01-2023_spacy_annotation_article_by_article_b')
os.makedirs(output_folder, exist_ok=True)


logging.info('Now generating file list')
input_folder = os.path.join('it', 'wikiextractor_output')
files = [os.path.join(root, f) for root, direc, files in os.walk(input_folder) for f in files]

logging.info('Now annotating Wikipedia!')

if __name__ == '__main__':
    
    with multiprocessing.Pool(processes=int(os.cpu_count()/3)) as p:
        p.map(preprocess, files)
    p.terminate()
    p.join()
