import googletrans

from googletrans import Translator
from tqdm import tqdm

from utils import read_ratings

words = list()
with open('data/fernandino_experiential_ratings.tsv') as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line =l.strip().split('\t')
        words.append(line[0])

'''
translator = Translator(
                    )
translations = translator.translate(words, origin='en', dest='de')

with open('data/fernandino_english_to_german.tsv', 'w') as o:
    o.write('original\tgoogle_translation\n')
    for translation in translations:
        #print(translation.text)
        o.write('{}\t{}\n'.format(translation.origin, translation.text))
'''

lancaster_ratings = read_ratings(hand=True) 
#splits = list(range(0, len(lancaster_ratings.keys()), 20000))
all_ws = [w for w in lancaster_ratings.keys()]+words 
with open('all_experiment_words_en.txt', 'w') as o:
    for w in all_ws:
        o.write('{}\n'.format(w))

'''
splits = [0, 20000]
all_trans = list()

for lang in ['de', 'it']:
    with tqdm() as counter:
        for i, start in enumerate(splits):
            if i == all_ws[-1]:
                curr_ws = all_ws[start:]
            else:
                curr_ws = all_ws[start:splits[i+1]]

            translator = Translator(
                                )
            translations = translator.translate(
                                                curr_ws,
                                                origin='en', 
                                                dest=lang,
                                            )
            for t in translations:
                all_trans.append((t.origin, t.text))
            counter.update(1)
    #with open('data/lancaster_english_to_german.tsv', 'w') as o:
    with open('data/en_to_{}.tsv'.format(lang), 'w') as o:
        o.write('original\tgoogle_translation\n')
        for o, t in all_trans:
            #print(translation.text)
            o.write('{}\t{}\n'.format(o, t))
'''
