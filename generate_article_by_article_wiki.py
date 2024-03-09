import argparse
import os
import re

def write_article(title, article, html_mapper, args, written):
    if len(title) < 200:
        ### short
        corr_title = re.sub('\W', r'_', title)
        os.makedirs(os.path.join(args.out, corr_title[:3]), exist_ok=True)
        print(corr_title)
        with open(os.path.join(args.out, corr_title[:3], '{}.tsv'.format(corr_title)), 'w') as o:
            for ar in article:
                ### correcting entity mentions
                ar = re.sub('&lt;a href="(.+?)"&gt;.+?/a&gt;', r' [[[\1]]] '.replace('\s', '_'), ar) 
                corr_ar = re.compile("|".join(html_mapper.keys())).sub(lambda ele: html_mapper[re.escape(ele.group(0))], ar)
                print(corr_ar)
                o.write('{}\n'.format(corr_ar))
                written = True
    return written

parser = argparse.ArgumentParser()
parser.add_argument('--wiki_folder', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()

### reading character_maps
html_mapper = dict()
with open('utf8_mapper.tsv') as i:
    for l in i:
        line = l.strip().split('\t')
        if len(line) != 3:
            continue
        if line[2][-1] == '%':
            exp = '%{}'.format(re.sub('\s', '', line[2])[:-1])
        else:
            exp = line[2]
        html_mapper[exp] = line[0]

os.makedirs(args.out, exist_ok=True)
written = False

article = list()
for direc in os.listdir(args.wiki_folder):
    for f in os.listdir(os.path.join(args.wiki_folder, direc)):
        with open(os.path.join(args.wiki_folder, direc, f)) as i:
            for l in i:
                line = l.strip()
                ### empty lines
                if len(line) < 5:
                    continue
                ### title
                if line[:6] == '</doc>':
                    continue
                if line[:8] == '<doc id=':
                    ### writing old
                    if len(article) >= 2:
                        written = write_article(title, article, html_mapper, args, written)
                    ### new
                    title = re.findall('title="(.+?)">', line)
                    print(title)
                    assert len(title) == 1
                    title = title[0]
                    article = list()
                    written = False
                else:
                    article.append(line)
    ### writing last article in file
    if not written:
        if len(article) >= 2:
            written = write_article(title, article, html_mapper, args, written)
