import googletrans

from googletrans import Translator

from utils import read_ratings

'''
words = list()
with open('data/fernandino_experiential_ratings.tsv') as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line =l.strip().split('\t')
        words.append(line[0])

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

translator = Translator(
                    )
translations = translator.translate([w for w in lancaster_ratings.keys()], origin='en', dest='de')
with open('data/lancaster_english_to_german.tsv', 'w') as o:
    o.write('original\tgoogle_translation\n')
    for translation in translations:
        #print(translation.text)
        o.write('{}\t{}\n'.format(translation.origin, translation.text))
