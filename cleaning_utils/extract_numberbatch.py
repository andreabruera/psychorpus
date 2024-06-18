import numpy
import os
import pickle

from tqdm import tqdm

vecs = dict()

with tqdm() as counter:
    with open('numberbatch-19.08.txt') as i:
        for l in i:
            line = l.strip().split('/')
            try:
                if line[2] == 'it':
                    w_v = line[3].split()
                    vecs[w_v[0]] = numpy.array(w_v[1:], dtype=numpy.float32)
                    counter.update(1)
            except IndexError:
                continue
with open(os.path.join('it', 'conceptnet_it.pkl'), 'wb') as o:
    pickle.dump(vecs, o)
