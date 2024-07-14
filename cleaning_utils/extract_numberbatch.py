import numpy
import os
import pickle

from tqdm import tqdm

lang = 'en'
base_folder = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'word_vectors')
assert os.path.exists(base_folder)

vecs = dict()

with tqdm() as counter:
    with open(os.path.join(base_folder, 'numberbatch-19.08.txt')) as i:
        for l in i:
            line = l.strip().split('/')
            try:
                if line[2] == lang:
                    w_v = line[3].split()
                    vecs[w_v[0]] = numpy.array(w_v[1:], dtype=numpy.float32)
                    counter.update(1)
            except IndexError:
                continue
with open(os.path.join(base_folder, lang, 'conceptnet_{}.pkl'.format(lang)), 'wb') as o:
    pickle.dump(vecs, o)
