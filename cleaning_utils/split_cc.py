import os
import re
import tqdm

from tqdm import tqdm

out_folder = '/import/dali-share-01/cc100-en'
os.makedirs(out_folder, exist_ok=True)
sents = list()
counter = 0

with open('cc100-en.txt') as i:
    for l in tqdm(i):
        line = l.strip()
        if line == '' or len(line.split())<5:
            continue
        sents.append(re.sub('\s+', r' ', line))
        if len(sents) == 100000:
            f_out = '{:010}.cc100'.format(counter)
            counter += 1
            with open(os.path.join(out_folder, f_out), 'w') as o:
                for s in sents:
                    o.write('{}\n'.format(s))
            sents = list()
if len(sents) > 0:
    f_out = '{:010}.cc100'.format(counter)
    with open(os.path.join(out_folder, f_out), 'w') as o:
        for s in sents:
            o.write('{}\n'.format(s))
