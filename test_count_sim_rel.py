import argparse
import fasttext
import numpy
import os
import pickle
import random
import scipy

from scipy import spatial, stats

from utils import build_ppmi_vecs, read_ratings, read_men, read_men_test, read_simlex

ratings = read_ratings()

men_sims = read_men()
#men_sims = read_men_test()
men_words = set([w for ws in men_sims.keys() for w in ws])
print('annotated MEN words: {}'.format(len(men_words)))

simlex_sims = read_simlex()
simlex_words = set([w for ws in simlex_sims.keys() for w in ws])
print('annotated SimLex999 words: {}'.format(len(simlex_words)))

test_words = men_words.union(simlex_words)
#test_words = men_words

for corpus in [
               #'bnc',
               'wac',
               #'wiki',
               #'opensubs',
               ]:
    if corpus == 'bnc':
        min_count = 10
    elif corpus in ['wac', 'wiki']:
        min_count = 100
    with open(os.path.join(
                           'pickles', 'en', corpus, 
                           'en_{}_uncased_word_freqs.pkl'.format(corpus),
                           ), 'rb') as i:
        freqs = pickle.load(i)
    with open(os.path.join(
                           'pickles', 'en', corpus, 
                           'en_{}_coocs_uncased_min_{}_win_20.pkl'.format(corpus, min_count),
                           ), 'rb') as i:
        coocs = pickle.load(i)
    with open(os.path.join(
                           'pickles', 'en', corpus, 
                           'en_{}_uncased_vocab_min_{}.pkl'.format(corpus, min_count),
                           ), 'rb') as i:
        vocab = pickle.load(i)
    missing = list()
    present = dict()
    for w in test_words:
        if w not in freqs.keys():
            missing.append(w)
        else:
            present[w] = freqs[w]
    print('\n')
    print('corpus: {}'.format(corpus))
    print('number of words missing: {}'.format(len(missing)))
    print('missing words:')
    print(missing)
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
    print(lower_ten_ws)
    print('number of items above 50 mentions: {}'.format(fifty_n))
    print('items with less than 50 mentions:')
    print(lower_fifty_ws)
    print('number of items above 100 mentions: {}'.format(hundred_n))
    print('items with less than 100 mentions:')
    print(lower_hundred_ws)
    print('average number of mentions: {}'.format(avg_n))
    print('median number of mentions: {}'.format(med_n))

    pruned_test_words = [w for w in test_words if w not in missing and vocab[w]!=0]

    ### removing rare words
    pruned_ratings = {w : dct for w, dct in ratings.items() if w in freqs.keys() and freqs[w] >= 100 and vocab[w]!=0}
    percent = int(len(pruned_ratings.items())*0.001)
    #percent = int(len(pruned_ratings.items())*0.05)
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
    ctx_words = sorted(ctx_words)
    ctx_idxs = [vocab[w] for w in ctx_words]
    vecs = {w : numpy.array([coocs[vocab[w]][idx] if idx in coocs[vocab[w]].keys() else 0 for idx in ctx_idxs]) for w in pruned_test_words}
    ### pmi
    ### building the PPMI matrix
    ### things are better when including in the rows the words from MEN...
    trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, ctx_words, ctx_words)
    #trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, test_men_words, ctx_words)
    ### using most frequent words
    freq_ctx_words = pruned_test_words + [w[0] for w in sorted(freqs.items(), key=lambda item: item[1], reverse=True) if vocab[w[0]]!=0][:4000]
    freq_pmi_vecs = build_ppmi_vecs(coocs, vocab, freq_ctx_words, freq_ctx_words)
    
    for case in [
                 'random',
                 'raw',
                 #'log2', 
                 'pmi',
                 'most_frequent_pmi',
                 'fasttext',
                 ]:
        if case == 'random':
            current_vecs = {k : numpy.array(random.sample(v.tolist(), k=len(v))) for k, v in vecs.items()}
        elif case == 'raw':
            current_vecs = {k : v for k, v in vecs.items()}
        elif case == 'log2':
            current_vecs = {k : numpy.log2(v+1) for k, v in vecs.items()}
        elif case == 'pmi':
            current_vecs = {k : trans_pmi_vecs[k] for k, v in trans_pmi_vecs.items()}
        elif case == 'most_frequent_pmi':
            current_vecs = {k : freq_pmi_vecs[k] for k, v in freq_pmi_vecs.items()}
        elif case == 'fasttext':
            ft = fasttext.load_model('../../dataset/word_vectors/en/cc.en.300.bin')
            current_vecs = {k : ft[k] for k, v in vecs.items()}
        for dataset_name, dataset in [
                                      ('MEN', men_sims),
                                      ('SimLex', simlex_sims),
                                      ]:
            ### creating word vectors
            print('removing test cases containing: {}'.format(missing))
            test_sims = dict()
            for ws, val in dataset.items():
                marker = True
                for w in ws:
                    if w not in pruned_test_words:
                        marker = False
                if marker:
                    test_sims[ws] = val
            real = list()
            pred = list()
            for k, v in test_sims.items():
                real.append(v)
                current_pred = 1 - scipy.spatial.distance.cosine(current_vecs[k[0]], current_vecs[k[1]])
                pred.append(current_pred)
            corr = scipy.stats.pearsonr(real, pred)
            print('\n')
            print('{} model'.format(case))
            print('correlation with {} dataset:'.format(dataset_name))
            print(corr)
