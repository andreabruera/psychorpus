import argparse
import fasttext
import gensim
import numpy
import os
import pickle
import random
import scipy

import gensim.downloader as api

from scipy import spatial, stats
from tqdm import tqdm

from utils import build_ppmi_vecs, read_ratings, read_men, read_men_test, read_simlex, read_fernandino

ratings = read_ratings()

for base_corpus in [
               #'bnc',
               #'wac',
               #'tagged_wiki',
               'opensubs',
               ]:
    print('\n\n{}'.format(base_corpus))
    min_count = 10
    #if corpus in ['bnc']:
    #elif corpus in ['wac', 'tagged_wiki','wiki', 'opensubs']:
    #    min_count = 10
    with open(os.path.join(
                           'pickles', 'en', base_corpus, 
                           'en_{}_uncased_word_freqs.pkl'.format(base_corpus),
                           ), 'rb') as i:
        base_freqs = pickle.load(i)
    with open(os.path.join(
                           'pickles', 'en', base_corpus, 
                           'en_{}_uncased_word_pos.pkl'.format(base_corpus),
                           ), 'rb') as i:
        base_pos = pickle.load(i)
    print('total size of the corpus: {} tokens'.format(sum(base_freqs.values())))
    with open(os.path.join(
                           'pickles', 'en', base_corpus, 
                           'en_{}_coocs_uncased_min_{}_win_20.pkl'.format(base_corpus, min_count),
                           #'en_{}_coocs_uncased_min_{}_win_4.pkl'.format(corpus, min_count),
                           ), 'rb') as i:
        base_coocs = pickle.load(i)
    with open(os.path.join(
                           'pickles', 'en', base_corpus, 
                           'en_{}_uncased_vocab_min_{}.pkl'.format(base_corpus, min_count),
                           ), 'rb') as i:
        base_vocab = pickle.load(i)
    fer_one_data, fer_one_words, fer_two_data, fer_two_words = read_fernandino(base_vocab, base_pos)

    test_words = set(fer_one_words).union(set(fer_two_words))
    present=dict()
    print('missing words:')
    missing = list()
    for w in test_words:
        if w not in base_freqs.keys():
            missing.append(w)
            print('{} - not appearing at all'.format(w))
            continue
        if base_vocab[w]==0:
            missing.append(w)
            print([m, base_pos[m]])
            continue
        else:
            present[w] = base_freqs[w]
    print('\n')
    print('corpus: {}'.format(base_corpus))
    print('number of words missing: {}'.format(len(missing)))
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

    print('total number of ratings words available: {}'.format(len(ratings.keys())))
    ### removing rare words
    pruned_test_words = [w for w in test_words if w not in missing]
    #freq_missing_ratings = set(ratings.keys()).difference(set(freqs.keys()))
    #voc_missing_ratings = [w for w, dct in ratings.items() if w in freqs.keys() and vocab[w]==0]
    #freq_missing_ratings = [w for w, dct in ratings.items() if w not in freqs.keys()]
    pruned_ratings = {w : dct for w, dct in ratings.items() if w in base_freqs.keys() and base_vocab[w]!=0}
    print('number of ratings words actually used: {}'.format(len(pruned_ratings.keys())))
    #percent = int(len(pruned_ratings.items())*0.001)
    #percent = int(len(pruned_ratings.items())*0.06)
    ### all words
    percent = int(len(pruned_ratings.items()))
    ### context words
    ### things improve when including the words directly
    ctx_words = set(pruned_test_words)
    #ctx_words = set()
    sem_dims = set([var for k,v in pruned_ratings.items() for var in v.keys()])
    for dim in sem_dims:
        sorted_ws = sorted([(w, v[dim]) for w, v in pruned_ratings.items()], key=lambda item: item[1])
        ctx_words = ctx_words.union(set([w for w, val in sorted_ws[-percent:]]))
        if dim == 'concreteness':
        #    continue
            ### also adding super abstract words
            ctx_words = ctx_words.union(set([w[0] for w in sorted_ws[:percent]]))
        #    ctx_words = ctx_words.union(set([w for w, val in sorted_ws[-one_percent:]]))
    print('considering {} context words'.format(len(ctx_words)))
    ctx_words = sorted(ctx_words)
    ### now adding counts from other corpora
    for other_corpus in [
               #'bnc',
               'wac',
               #'tagged_wiki',
               #'opensubs',
               ]:
        assert base_corpus != other_corpus
        print('\n\n{}'.format(other_corpus))
        with open(os.path.join(
                               'pickles', 'en', other_corpus, 
                               'en_{}_uncased_word_freqs.pkl'.format(other_corpus),
                               ), 'rb') as i:
            other_freqs = pickle.load(i)
        print('total size of the corpus: {} tokens'.format(sum(other_freqs.values())))
        with open(os.path.join(
                               'pickles', 'en', other_corpus, 
                               'en_{}_coocs_uncased_min_{}_win_20.pkl'.format(other_corpus, min_count),
                               #'en_{}_coocs_uncased_min_{}_win_4.pkl'.format(other_corpus, min_count),
                               ), 'rb') as i:
            other_coocs = pickle.load(i)
        with open(os.path.join(
                               'pickles', 'en', other_corpus, 
                               'en_{}_uncased_vocab_min_{}.pkl'.format(other_corpus, min_count),
                               ), 'rb') as i:
            other_vocab = pickle.load(i)
        print('now adding counts from {}'.format(other_corpus))
        for w_i, w_one in tqdm(enumerate(ctx_words)):
            for w_two_i, w_two in enumerate(ctx_words):
                if w_two_i < w_i:
                    continue
                try:
                    base_coocs[base_vocab[w_one]][base_vocab[w_two]] += other_coocs[other_vocab[w_one]][other_vocab[w_two]]
                except KeyError:
                    continue

    ctx_idxs = [base_vocab[w] for w in ctx_words]
    vecs = {w : numpy.array([base_coocs[base_vocab[w]][idx] if idx in base_coocs[base_vocab[w]].keys() else 0 for idx in ctx_idxs]) for w in pruned_test_words}
    ### pmi
    ### building the PPMI matrix
    ### things are better when including in the rows the words from the dataset...
    trans_pmi_vecs = build_ppmi_vecs(base_coocs, base_vocab, ctx_words, ctx_words)
    #trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, test_men_words, ctx_words)
    ### using most frequent words
    freq_ctx_words = pruned_test_words + [w[0] for w in sorted(base_freqs.items(), key=lambda item: item[1], reverse=True) if base_vocab[w[0]]!=0][:4000]
    freq_pmi_vecs = build_ppmi_vecs(base_coocs, base_vocab, freq_ctx_words, freq_ctx_words)
    
    for case in [
                 #'random',
                 #'raw',
                 #'log2', 
                 'pmi',
                 #'most_frequent_pmi',
                 #'fasttext',
                 'glove',
                 'word2vec',
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
        elif case == 'glove':
            model = api.load("glove-wiki-gigaword-300")
            current_vecs = {k : model[k] for k, v in vecs.items()}
            del model
        elif case == 'word2vec':
            model = api.load("word2vec-google-news-300")
            current_vecs = {k : model[k.replace('axe', 'ax')] for k, v in vecs.items()}
            del model
        for dataset_name, dataset, dataset_words in [
                                      ('fernandino_1', fer_one_data, fer_one_words),
                                      ('fernandino_2', fer_two_data, fer_two_words),
                                      ]:
            ### creating word vectors sim matrices
            ### RSA
            ### removing diagonal
            pred = [1 - scipy.spatial.distance.cosine(current_vecs[k_one], current_vecs[k_two]) for k_one_i, k_one in enumerate(dataset_words) for k_two_i, k_two in enumerate(dataset_words) if k_two_i>k_one_i]
            ### leaving diagonal
            #pred = [1 - scipy.spatial.distance.cosine(current_vecs[k_one], current_vecs[k_two]) for k_one_i, k_one in enumerate(dataset_words) for k_two_i, k_two in enumerate(dataset_words) if k_two_i>=k_one_i]
            results = list()
            for subject, subject_data in dataset.items():
                #corr = scipy.stats.pearsonr(subject_data, pred)
                corr = scipy.stats.spearmanr(subject_data, pred)
                results.append(corr.statistic)
            print('\n')
            print('{} model'.format(case))
            print('correlation with {} dataset:'.format(dataset_name))
            print(numpy.average(results))
