import fasttext
import numpy
import os
import pickle
import scipy

from scipy import spatial

from utils import build_ppmi_vecs, read_ratings
from utf_utils import transform_german_word

### loading german
ft = fasttext.load_model('../../dataset/word_vectors/de/cc.de.300.bin')
### loading frequencies
with open(os.path.join(
                       'pickles', 'de', 'joint', 
                       'de_joint_uncased_word_freqs.pkl',
                       ), 'rb') as i:
    freqs = pickle.load(i)
### loading pos
with open(os.path.join(
                       'pickles', 'de', 'joint', 
                       'de_joint_uncased_word_pos.pkl',
                       ), 'rb') as i:
    pos = pickle.load(i)
### vocab
with open(os.path.join(
                       'pickles', 'de', 'joint', 
                       'de_joint_uncased_vocab_min_10.pkl',
                       ), 'rb') as i:
    vocab = pickle.load(i)
### coocs
with open(os.path.join(
                       'pickles', 'de', 'joint', 
                       'de_joint_coocs_uncased_min_10_win_4.pkl',
                       ), 'rb') as i:
    coocs = pickle.load(i)

uttered_words = list()

with open(os.path.join(
                       'data', 
                       'german_ifg.tsv'
                       )) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            continue
        task = line[header.index('task')]
        if task != 'sem':
            continue
        word = line[header.index('utterance')].strip()
        if word == 'NA':
            continue
        w_ones = transform_german_word(word)
        w_ones = [w for w in w_ones if w in vocab.keys() and w in freqs.keys() and vocab[w]!=0]
        if len(w_ones) == 0:
            continue
        uttered_words.extend(w_ones)
        category = line[header.index('item')].split('.')[0]
        w_twos = transform_german_word(category)
        w_twos = [w for w in w_twos if w in vocab.keys() and w in freqs.keys() and vocab[w]!=0]
        print(w_twos)
        assert len(w_twos) > 0
        uttered_words.extend(w_twos)

ratings = read_ratings(hand=True)
test_words = [w for w in ratings.keys()]
trans = dict()
### reading translations
inv_trans = dict()
to_write = list()
fs = [
        #'en_to_de.tsv'
        'lancaster_english_to_german.tsv',
        'fernandino_english_to_german.tsv',
        ]
for f in fs:
    with open(os.path.join('data', f)) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.lower().strip().split('\t')
            trans[line[0].strip()] = line[1].strip().replace('-', '_').lower()
            inv_trans[line[1].strip().replace('-', '_').lower()] = line[0].strip()
            to_write.append((line[0].strip(), line[1].strip().replace('-', '_').lower()))
with open(os.path.join('data', 'en_to_de.tsv'), 'w') as o:
    o.write('english\tgerman\n')
    for k, v in to_write:
        o.write('{}\t{}\n'.format(k, v))
missing = list()
present = dict()
for w in test_words:
    try:
        w = trans[w].lower()
        #if inv_trans[trans[w].lower()] != w:
        #    missing.append(w)
        #    continue
    except KeyError:
        missing.append(w)
        continue
    if w not in freqs.keys():
        missing.append(w)
        print('{} - not appearing at all'.format(w))
        continue
    if w not in vocab.keys() or vocab[w]==0:
        missing.append(w)
        continue
    if vocab[w] not in coocs.keys():
        missing.append(w)
        print([w, freqs[w]])
        continue
    else:
        present[w] = freqs[w]
print('\n')
print('number of words missing: {}'.format(len(missing)))
print('missing words:')
#print(missing)
min_n = min(list(present.values()))
max_n = max(list(present.values()))
lower_ten_ws = [w for w, val in present.items() if val < 10]
lower_fifty_ws = [w for w, val in present.items() if val < 50]
lower_hundred_ws = [w for w, val in present.items() if val < 100]
ten_n = sum([1 for val in present.values() if val >= 10])
fifty_n = sum([1 for val in present.values() if val >= 50])
hundred_n = sum([1 for val in present.values() if val >= 100])
avg_n = numpy.average(list(present.values()))
med_n = numpy.median(list(present.values()))
print('minimum number of mentions: {}'.format(min_n))
print('maximum number of mentions: {}'.format(max_n))
print('number of items above 10 mentions: {}'.format(ten_n))
print('items with less than 10 mentions:')
#print(lower_ten_ws)
print('number of items above 50 mentions: {}'.format(fifty_n))
print('items with less than 50 mentions:')
#print(lower_fifty_ws)
print('number of items above 100 mentions: {}'.format(hundred_n))
print('items with less than 100 mentions:')
#print(lower_hundred_ws)
print('average number of mentions: {}'.format(avg_n))
print('median number of mentions: {}'.format(med_n))

#pruned_test_words = [w for w in uttered_words if w not in missing and vocab[w]!=0]
pruned_test_words = [w for w in present.keys()]
pruned_ratings = {trans[w] : dct for w, dct in ratings.items() if trans[w] in present.keys()}
        #trans.keys() and trans[w] in freqs.keys() and vocab[trans[w]]!=0 and vocab[trans[w]] in coocs.keys()}

### removing rare words
#pruned_ratings = {w : dct for w, dct in ratings.items() if w in freqs.keys() and vocab[w]!=0}
#percent = int(len(pruned_ratings.items())*0.001)
#percent = int(len(pruned_ratings.items())*0.05)
percent = int(len(pruned_ratings.items())*1)
### context words
### things improve when including the words directly
ctx_words = set(pruned_test_words)
#ctx_words = set()
sem_dims = set([var for k,v in pruned_ratings.items() for var in v.keys()])
for dim in sem_dims:
    if dim == 'concreteness':
        continue
    sorted_ws = sorted([(w, v[dim]) for w, v in pruned_ratings.items()], key=lambda item: item[1])
    ctx_words = ctx_words.union(set([w for w, val in sorted_ws[-percent:]]))
    ### also adding super abstract words
    #    ctx_words = ctx_words.union(set([w[0] for w in sorted_ws[:one_percent]]))
    #    ctx_words = ctx_words.union(set([w for w, val in sorted_ws[-one_percent:]]))
print('considering {} context words'.format(len(ctx_words)))
ctx_words = sorted(ctx_words.union(set(uttered_words)))
ctx_idxs = [vocab[w] for w in ctx_words]
vecs = {w : numpy.array([coocs[vocab[w]][idx] if idx in coocs[vocab[w]].keys() else 0 for idx in ctx_idxs]) for w in pruned_test_words}
### pmi
### building the PPMI matrix
trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, ctx_words, ctx_words, smoothing=False)

### german IFG
#uttered_words = set(uttered_words)
with open('german_ifg_r.tsv', 'w') as o:
    o.write('sub\t')
    o.write('session\t')
    o.write('condition\t')
    o.write('category\t')
    o.write('word\t')
    o.write('rt\t')
    o.write('log_rt\t')
    o.write('ft_distance\t')
    o.write('ppmi_distance\t')
    o.write('w_one_frequency\t')
    o.write('w_one_log10_frequency\t')
    o.write('w_one_length\t')
    o.write('w_two_frequency\t')
    o.write('w_two_log10_frequency\t')
    o.write('w_two_length\t')
    o.write('\n')
    with open(os.path.join(
                           'data', 
                           'german_ifg.tsv'
                           )) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                header = line.copy()
                continue
            task = line[header.index('task')]
            if task != 'sem':
                continue
            word = line[header.index('utterance')].strip()
            if word == 'NA':
                continue
            w_ones = transform_german_word(word)
            #print(w_ones)
            w_ones = [w for w in w_ones if w in freqs.keys() and vocab[w]!=0]
            if len(w_ones) == 0:
                continue
            freq_w_one = sum([freqs[w] for w in w_ones])
            len_w_one = len(word)
            #if word.lower() not in vocab.keys() or vocab[word]==0:
            #    print(word)
            sub = line[header.index('subj')]
            category = line[header.index('item')].split('.')[0]
            w_twos = transform_german_word(category)
            w_twos = [w for w in w_twos if w in freqs.keys() and vocab[w]!=0]
            assert len(w_twos) > 0
            freq_w_two = sum([freqs[w] for w in w_twos])
            len_w_two = len(category)
            session = line[header.index('session')]
            condition = line[header.index('stim')]
            ### dummy coding, reference at 1.
            if condition == 'vertex':
                condition = 1.
            elif condition == 'aIFG':
                condition = 1.5
            elif condition == 'pIFG':
                condition = 0.5
            else:
                raise RuntimeError()
            rt = float(line[header.index('RT')].replace(',', '.'))
            ### rt
            log_rt = numpy.log(1+rt)
            ### ft
            combs = list()
            for w_one in w_ones:
                for w_two in w_twos:
                    combs.append((w_one, w_two))
            ft_distance = list()
            ppmi_distance = list()
            for w_one, w_two in combs:
                ft_distance.append(scipy.spatial.distance.cosine(ft[w_one], ft[w_two]))
                ppmi_distance.append(scipy.spatial.distance.cosine(trans_pmi_vecs[w_one], trans_pmi_vecs[w_two]))
            ft_distance = numpy.average(ft_distance)
            ppmi_distance = numpy.average(ppmi_distance)
            o.write('{}\t'.format(sub))
            o.write('{}\t'.format(session))
            o.write('{}\t'.format(condition))
            o.write('{}\t'.format(category))
            o.write('{}\t'.format(word))
            o.write('{}\t'.format(rt))
            o.write('{}\t'.format(log_rt))
            o.write('{}\t'.format(ft_distance))
            o.write('{}\t'.format(ppmi_distance))
            o.write('{}\t'.format(freq_w_one))
            o.write('{}\t'.format(numpy.log10(freq_w_one)))
            o.write('{}\t'.format(len(word)))
            o.write('{}\t'.format(freq_w_two))
            o.write('{}\t'.format(numpy.log10(freq_w_two)))
            o.write('{}\t'.format(len(category)))
            o.write('\n')
            #print(word)
