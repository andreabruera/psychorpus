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

from utils import build_ppmi_vecs, read_brys_ratings, read_exp48, read_ratings, read_men, read_men_test, read_simlex, read_fernandino

print('now loading pre-trained models...')
#w2v = api.load("word2vec-google-news-300")
#glove = api.load("glove-wiki-gigaword-300")
#ft = fasttext.load_model('../../dataset/word_vectors/en/cc.en.300.bin')
print('loaded!')
ratings = read_ratings()

brys_ratings = read_brys_ratings()

for corpus in [
               'opensubs',
               #'tagged_wiki',
               #'bnc',
               #'wac',
               ]:
    for min_count in [
                      10, 
                      #20,
                      #50,
                      #100,
                      ]:
        for percent_val in [
                            #0.001, 
                            #0.01, 
                            #0.1, 
                            1,
                            ]:
            for win_size in [
                             4, 
                             #20,
                             ]:
                print('\n\n{} - min {} - {} percent - window size {}'.format(corpus, min_count, percent_val*100, win_size))
                with open(os.path.join(
                                       'pickles', 'en', corpus, 
                                       'en_{}_uncased_word_freqs.pkl'.format(corpus),
                                       ), 'rb') as i:
                    freqs = pickle.load(i)
                with open(os.path.join(
                                       'pickles', 'en', corpus, 
                                       'en_{}_uncased_word_pos.pkl'.format(corpus),
                                       ), 'rb') as i:
                    pos = pickle.load(i)
                print('total size of the corpus: {} tokens'.format(sum(freqs.values())))
                try:
                    with open(os.path.join(
                                           'pickles', 'en', corpus, 
                                           'en_{}_coocs_uncased_min_{}_win_{}.pkl'.format(corpus, min_count, win_size),
                                           #'en_{}_coocs_uncased_min_{}_win_4.pkl'.format(corpus, min_count),
                                           ), 'rb') as i:
                        coocs = pickle.load(i)
                except FileNotFoundError:
                    print('this model is missing!')
                    continue
                with open(os.path.join(
                                       'pickles', 'en', corpus, 
                                       'en_{}_uncased_vocab_min_{}.pkl'.format(corpus, min_count),
                                       ), 'rb') as i:
                    vocab = pickle.load(i)
                
                words, data = read_fernandino(vocab, pos)
                exp48, exp48_words = read_exp48(words)
                sm = [
                      'taste',
                      'smell',
                      'audition',
                      'vision',
                      'touch',
                      'hand',
                      'foot',
                      'mouth',
                      ]
                idxs = [w_i for w_i, w in enumerate(exp48_words) if w in sm]
                assert len(idxs) == 8
                sm8 = {k : v[idxs] for k, v in exp48.items()}

                test_words = set(words[1]).union(set(words[2]))
                present=dict()
                print('missing words:')
                missing = list()
                for w in test_words:
                    if w not in freqs.keys():
                        missing.append(w)
                        print('{} - not appearing at all'.format(w))
                        continue
                    if vocab[w]==0:
                        missing.append(w)
                        print([m, pos[m]])
                        continue
                    else:
                        present[w] = freqs[w]
                print('\n')
                print('corpus: {}'.format(corpus))
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
                pruned_ratings = {w : dct for w, dct in ratings.items() if w in freqs.keys() and vocab[w]!=0}
                print('number of ratings words actually used: {}'.format(len(pruned_ratings.keys())))
                #percent = int(len(pruned_ratings.items())*0.001)
                #percent = int(len(pruned_ratings.items())*0.06)
                percent = int(len(pruned_ratings.items())*percent_val)
                ### all words
                #percent = int(len(pruned_ratings.items()))
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
                ctx_idxs = [vocab[w] for w in ctx_words]
                vecs = {w : numpy.array([coocs[vocab[w]][idx] if idx in coocs[vocab[w]].keys() else 0 for idx in ctx_idxs]) for w in pruned_test_words}
                ### pmi
                ### building the PPMI matrix
                ### things are better when including in the rows the words from MEN...
                trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, ctx_words, ctx_words)
                #trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, test_men_words, ctx_words)
                '''
                ### using most frequent words
                freq_ctx_words = pruned_test_words + [w[0] for w in sorted(freqs.items(), key=lambda item: item[1], reverse=True) if vocab[w[0]]!=0][:len(ctx_idxs)]
                freq_pmi_vecs = build_ppmi_vecs(coocs, vocab, freq_ctx_words, freq_ctx_words)
                '''
                for mode in [
                            'averaged_sub_mtrxs', 
                            'individual_sub_mtrxs',
                            ]:
                    if mode == 'averaged_sub_mtrxs':
                        words, data = read_fernandino(vocab, pos, avg_subjects=True)
                    elif mode == 'individual_sub_mtrxs':
                        words, data = read_fernandino(vocab, pos, avg_subjects=False)
                    #exp48_ctx_words = pruned_test_words + exp48_words
                    #exp48_pmi_vecs = build_ppmi_vecs(coocs, vocab, exp48_ctx_words, exp48_ctx_words)
                    
                    for case in [
                                 #'random',
                                 #'raw',
                                 #'log2', 
                                 #'pmi',
                                 'pmi75',
                                 #'most_frequent_pmi',
                                 #'fasttext',
                                 #'exp48',
                                 #'exp_sm8',
                                 #'pmi_exp48',
                                 #'glove',
                                 #'word2vec',
                                 #'word_length',
                                 #'raw_freqs',
                                 #'log_freqs',
                                 #'arousal', 
                                 #'dominance',
                                 #'valence',
                                 #'aoa',
                                 #'concreteness',
                                  #'taste',
                                  #'smell',
                                  #'audition',
                                  #'vision',
                                  #'touch',
                                  #'hand',
                                 ]:
                        if case == 'random':
                            current_vecs = {k : numpy.array(random.sample(v.tolist(), k=len(v))) for k, v in vecs.items()}
                        elif case == 'word_length':
                            current_vecs = {k : len(k) for k, v in vecs.items()}
                        elif case in ['arousal', 'dominance', 'valence', 'concreteness', 'aoa']:
                            current_vecs = {k : brys_ratings[case][k] for k in pruned_test_words}
                        elif case in [
                                  'taste',
                                  'smell',
                                  'audition',
                                  'vision',
                                  'touch',
                                  'hand',
                                ]:
                            current_vecs = {k : v[exp48_words.index(case)] for k, v in exp48.items()}
                        elif case == 'raw_freqs':
                            current_vecs = {k : freqs[k] for k, v in vecs.items()}
                        elif case == 'log_freqs':
                            current_vecs = {k : numpy.log10(freqs[k]) for k, v in vecs.items()}
                        elif case == 'raw':
                            current_vecs = {k : v for k, v in vecs.items()}
                        elif case == 'log2':
                            current_vecs = {k : numpy.log2(v+1) for k, v in vecs.items()}
                        elif case == 'pmi':
                            current_vecs = {k : trans_pmi_vecs[k] for k, v in trans_pmi_vecs.items()}
                        elif case == 'pmi75':
                            current_vecs = {k : trans_pmi_vecs[k] for k, v in trans_pmi_vecs.items()}
                        elif case == 'pmi_exp48':
                            current_vecs = {k : exp48_pmi_vecs[k] for k, v in exp48_pmi_vecs.items()}
                        elif case == 'exp_sm8':
                            current_vecs = {k : sm8[k] for k, v in sm8.items()}
                        elif case == 'exp48':
                            current_vecs = {k : exp48[k] for k, v in exp48.items()}
                        elif case == 'most_frequent_pmi':
                            current_vecs = {k : freq_pmi_vecs[k] for k, v in freq_pmi_vecs.items()}
                        elif case == 'fasttext':
                            current_vecs = {k : ft[k] for k, v in vecs.items()}
                        elif case == 'glove':
                            current_vecs = {k : glove[k] for k, v in vecs.items()}
                        elif case == 'word2vec':
                            current_vecs = {k : w2v[k.replace('axe', 'ax')] for k, v in vecs.items()}
                        #for dataset_name, dataset, dataset_words in [
                        #                              ('fernandino_1_language_network', fer_one_lang_data, fer_one_lang_words),
                        #                              ('fernandino_2_language_network', fer_two_lang_data, fer_two_lang_words),
                        #                              ]:
                        for dataset_number, dataset_words in words.items():
                            results = os.path.join(
                                                   'results', 
                                                   'fernandino{}'.format(dataset_number),
                                                   'full_dataset',
                                                   mode
                                                   )
                            os.makedirs(
                                        results, 
                                        exist_ok=True,
                                        )
                            if 'pmi' in case:
                                case = '{}_{}_min{}_{}_win{}'.format(
                                                          case, 
                                                          corpus, 
                                                          min_count, 
                                                          percent_val, 
                                                          win_size,
                                                          )
                            with open(
                                  os.path.join(
                                      results,  
                                      '{}.txt'.format(
                                               case, 
                                               )
                                      ), 'w') as o:
                                for brain_area, dataset in data[dataset_number].items():
                                    ### creating word vectors sim matrices
                                    ### RSA
                                    ### removing diagonal
                                    if case in ['word_length', 'raw_freqs', 'log_freqs'] + sm:
                                        pred = [-abs(current_vecs[k_one]-current_vecs[k_two]) for k_one_i, k_one in enumerate(dataset_words) for k_two_i, k_two in enumerate(dataset_words) if k_two_i>k_one_i]
                                    else:
                                        pred = [1 - scipy.spatial.distance.cosine(current_vecs[k_one], current_vecs[k_two]) for k_one_i, k_one in enumerate(dataset_words) for k_two_i, k_two in enumerate(dataset_words) if k_two_i>k_one_i]
                                    ### leaving diagonal
                                    #pred = [1 - scipy.spatial.distance.cosine(current_vecs[k_one], current_vecs[k_two]) for k_one_i, k_one in enumerate(dataset_words) for k_two_i, k_two in enumerate(dataset_words) if k_two_i>=k_one_i]
                                    results = list()
                                    for subject, subject_data in dataset.items():
                                        #corr = scipy.stats.pearsonr(subject_data, pred)
                                        corr = scipy.stats.spearmanr(subject_data, pred)
                                        results.append(corr.statistic)
                                    print('\n')
                                    print(brain_area)
                                    print('{} model'.format(case))
                                    print('\n')
                                    o.write('{}\t'.format(brain_area))
                                    print('correlation with {} - fernandino {} dataset:'.format(brain_area, dataset_number))
                                    #o.write('correlation with {} - fernandino {} dataset:'.format(brain_area, dataset_number))
                                    #o.write('\n')
                                    for r in results:
                                        o.write('{}\t'.format(float(r)))
                                    print(numpy.average(results))
                                    #o.write('{}'.format(numpy.average(results)))
                                    o.write('\n')
