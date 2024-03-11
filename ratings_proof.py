import argparse
import numpy
import os
import pickle

from utils import read_ratings

ratings = read_ratings()
print('annotated words: {}'.format(len(ratings.keys())))

top_auditory = sorted([(w, v['auditory']) for w, v in ratings.items()], key=lambda item: item[1])
import pdb; pdb.set_trace()

for corpus in [
               'bnc',
               'wac',
               #'wiki',
               #'opensubs',
               ]:
    with open(os.path.join(
                           'pickles', 'en', corpus, 
                           'en_{}_uncased_word_freqs.pkl'.format(corpus),
                           ), 'rb') as i:
        freqs = pickle.load(i)
    missing = list()
    present = dict()
    for w in ratings.keys():
        if w not in freqs.keys():
            missing.append(w)
        else:
            present[w] = freqs[w]
    print('\n')
    print('corpus: {}'.format(corpus))
    print('number of words missing: {}'.format(len(missing)))
    min_n = min(list(present.values()))
    max_n = max(list(present.values()))
    ten_n = sum([1 for val in present.values() if val >= 10])
    fifty_n = sum([1 for val in present.values() if val >= 50])
    hundred_n = sum([1 for val in present.values() if val >= 100])
    avg_n = numpy.average(list(present.values()))
    med_n = numpy.median(list(present.values()))
    print('minimum number of mentions: {}'.format(min_n))
    print('maximum number of mentions: {}'.format(max_n))
    print('number of items above 10 mentions: {}'.format(ten_n))
    print('number of items above 50 mentions: {}'.format(fifty_n))
    print('number of items above 100 mentions: {}'.format(hundred_n))
    print('average number of mentions: {}'.format(avg_n))
    print('median number of mentions: {}'.format(med_n))
