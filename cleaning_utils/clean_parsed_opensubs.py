import argparse
import os
import re

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--language',
                    choices=[
                             'en',
                             'it',
                             'de',
                             ],
                    required=True,
                    )
args = parser.parse_args()

in_folder = os.path.join(
                         '..',
                         '..',
                         'dataset',
                         'corpora',
                         args.language,
                         'OpenSubtitles',
                         'parsed',
                         args.language,
                         )
assert os.path.exists(in_folder)

out_folder = os.path.join(
                         '..',
                         '..',
                         'dataset',
                         'corpora',
                         args.language,
                         'opensubs-2018_parsed_{}'.format(args.language),
                         )
os.makedirs(out_folder, exist_ok=True)
for year in tqdm(os.listdir(in_folder)):
    out_year = os.path.join(out_folder, year)
    os.makedirs(out_year, exist_ok=True)
    for other_f in os.listdir(os.path.join(in_folder, year)):
        for f in os.listdir(os.path.join(in_folder, year, other_f)):
            if 'xml' not in f:
                continue
            with open(os.path.join(in_folder, year, other_f, f)) as i:
                out_f = os.path.join(out_year, f)
                if os.path.exists(out_f):
                    out_f = '{}#'.format(out_f)
                with open(out_f, 'w') as o:
                    o.write('word\tlemma\tpos\tdep\n')
                    for l in i:
                        pos = re.findall('(?<=upos=")(.+?)(?=")', l)
                        dep = re.findall('(?<=deprel=")(.+?)(?=")', l)
                        lemma = re.findall('(?<=lemma=")(.+?)(?=")', l)
                        word = re.findall('(?<=>)(.+?)(?=</w>)', l)
                        if len(pos) == 1 and len(dep) == 1 and len(lemma) == 1 and len(word) == 1:
                            ### all good
                            o.write('{}\t{}\t{}\t{}\n'.format(word[0], lemma[0], pos[0], dep[0]))
                            print(word[0])
                        elif l.strip() == '</s>':
                            o.write('<EOS>\t<EOS>\t<EOS>\t<EOS>\n')
