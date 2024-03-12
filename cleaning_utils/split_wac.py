import argparse
import os

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--language', choices=['en', 'it', 'de'], required=True)
parser.add_argument('--corpora_folder', required=True)
args = parser.parse_args()

assert os.path.join(args.corpora_folder)

if args.language == 'en':
    wac_folder = 'PukWaC'
    encoding = 'latin-1'
elif args.language == 'it':
    wac_folder = 'itwac'
    encoding = 'latin-1'
elif args.language == 'de':
    wac_folder = 'sdewac-v3-tagged'
    encoding = 'latin-1'

full_wac_folder = os.path.join(args.corpora_folder, args.language, wac_folder)
assert os.path.exists(full_wac_folder)

out_folder = os.path.join(args.language, '{}_smaller_files'.format(wac_folder))
os.makedirs(out_folder, exist_ok=True)

current_file = list()
n_sentences = 0
file_counter = 0
with tqdm() as general_counter:
    for f in os.listdir(full_wac_folder):
        if 'wac' in f:
            with open(os.path.join(full_wac_folder, f), encoding=encoding) as i:
                for l in i:
                    current_file.append(l.strip())
                    if l.strip()[:4] == '</s>':
                        n_sentences += 1
                    if n_sentences == 100000:
                        with open(os.path.join(out_folder, '{}_wac_{:04}.tsv'.format(args.language, file_counter)), 'w') as o:
                            for current_l in current_file:
                                o.write('{}\n'.format(current_l))
                        current_file = list()
                        n_sentences = 0
                        file_counter += 1
                        general_counter.update(1)

with open(os.path.join(out_folder, '{}_wac_{:08}.tsv'.format(args.language, file_counter)), 'w') as o:
    for current_l in current_file:
        o.write('{}\n'.format(current_l))
