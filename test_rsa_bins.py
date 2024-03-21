import numpy
import os
import pickle
import random
import scipy
import tqdm

from tqdm import tqdm

from utils import build_ppmi_vecs, read_ratings, read_men, read_men_test, read_simlex, read_fernandino

def bins_rsa_test(words, ratings, model, brain_data, out_file):

    n_items = 32
    n_items = 8
    dataset_results = dict()
    relevant_keys = ratings[words[0]].keys()

    bins = {'{}_{}'.format(case, i) : list() for i in range(5) for case in relevant_keys}

    for case_i, case in enumerate(relevant_keys):
        print(case)
        counter = 0
        #for beg, end in tqdm([(0., 1.), (1, 2), (2, 3), (3, 4), (4, 5)]):
        for beg, end in tqdm([(0., .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.)]):
            bin_words = [w for w in words if w in ratings.keys() and ratings[w][case]>=beg and ratings[w][case]<=end]
            if len(bin_words) < n_items:
                print('nan')
                bins['{}_{}'.format(case, counter)] = [numpy.nan for s in brain_data.keys()]
                counter += 1
                continue
            bin_results = list()
            for _ in range(100):
                iter_results = list()
                current_bin_words = random.sample(bin_words, k=n_items)
                #current_bin_idxs = [ctx_words.index(w) for w in current_bin_words]

                #sim_model = numpy.array([[1 - scipy.spatial.distance.cosine(vecs[k], vecs[k_two]) for k_two_i, k_two in enumerate(current_bin_words) if k_two_i>k_i] for k_i, k in enumerate(current_bin_words)]).flatten()
                #sim_model = numpy.array([1 - scipy.spatial.distance.cosine(vecs[k], vecs[k_two]) for k_two_i, k_two in enumerate(current_bin_words) if k_two_i>k_i] for k_i, k in enumerate(current_bin_words)])
                #data = {k : numpy.array([[vec[i] for i_i, i in enumerate(current_bin_idxs) if all_words[i]!=k] for vec in v]) for k, v in all_data.items() if k in current_bin_words}
                sim_model = [1-scipy.spatial.distance.cosine(vecs[w_one], vecs[w_two]) for w_one_i, w_one in enumerate(current_bin_words) for w_two_i, w_two in enumerate(current_bin_words) if w_two_i>w_one_i]
                for s, s_data in brain_data.items():
                    sim_brain = [s_data[tuple(sorted([w_one, w_two]))] for w_one_i, w_one in enumerate(current_bin_words) for w_two_i, w_two in enumerate(current_bin_words) if w_two_i>w_one_i]
                    corr = scipy.stats.spearmanr(sim_model, sim_brain)[0]
                    iter_results.append(corr)
                bin_results.append(iter_results)
            bin_results = numpy.average(bin_results, axis=0)
            #print(bin_results)
            bins['{}_{}'.format(case, counter)] = bin_results
            counter += 1

    with open(out_file, 'w') as o:
        o.write('bin\tresults\n')
        for k, v in bins.items():
            assert len(v) == len(brain_data.keys())
            #print(len(v))
            o.write('{}\t'.format(k))
            for val in v:
                o.write('{}\t'.format(val))
            o.write('\n')

### reading files
corpus = [
               'opensubs',
               #'tagged_wiki',
               #'bnc',
               #'wac',
               ][0]
min_count = [
                  10, 
                  #100,
                  ][0]
percent_val = [
                    #0.001, 
                    #0.01, 
                    #0.1, 
                    1,
                    ][0]
win_size = [
                 4, 
                 #20,
                 ][0]
with open(os.path.join(
                       'pickles', 'en', corpus, 
                       'en_{}_uncased_word_pos.pkl'.format(corpus),
                       ), 'rb') as i:
    pos = pickle.load(i)
with open(os.path.join(
                       'pickles', 'en', corpus, 
                       'en_{}_uncased_vocab_min_{}.pkl'.format(corpus, min_count),
                       ), 'rb') as i:
    vocab = pickle.load(i)
words, _, data = read_fernandino(vocab, pos, return_dict=True)

with open(os.path.join(
                       'pickles', 'en', corpus, 
                       'en_{}_coocs_uncased_min_{}_win_{}_pmi_rows_cols.pkl'.format(corpus, min_count, win_size),
                       ), 'rb') as i:
    ctx_words = pickle.load(i)
ratings = {k : v for k, v in read_ratings(hand=True).items() if k in ctx_words}
mtrxs = dict()
with open(os.path.join(
                       'pickles', 'en', corpus, 
                       'en_{}_coocs_uncased_min_{}_win_{}_pmi.pkl'.format(corpus, min_count, win_size),
                       ), 'rb') as i:
    pmi_mtrx = pickle.load(i)
with open(os.path.join(
                       'pickles', 'en', corpus, 
                       'en_{}_coocs_uncased_min_{}_win_{}_rand.pkl'.format(corpus, min_count, win_size),
                       ), 'rb') as i:
    mtrxs['rand'] = pickle.load(i)
### powers
### power + shuffle
for power in [0.05, 0.1, 0.25, 0.5, 0.75]:
    with open(os.path.join(
                           'pickles', 'en', corpus, 
                           'en_{}_coocs_uncased_min_{}_win_{}_rand_pow_{}.pkl'.format(corpus, min_count, win_size, power),
                           ), 'rb') as i:
        mtrxs['rand_pow_{}'.format(power)] = pickle.load(i)
### power only
for power in [0.05, 0.1, 0.25, 0.5, 0.75]:
    with open(os.path.join(
                           'pickles', 'en', corpus, 
                           'en_{}_coocs_uncased_min_{}_win_{}_pow_{}.pkl'.format(corpus, min_count, win_size, power),
                           ), 'rb') as i:
        mtrxs['pow_{}'.format(power)] = pickle.load(i)

out = os.path.join('results', 'rsa', 'bins')
os.makedirs(out, exist_ok=True)

selected_areas = [
                  '_semantic_network',
                  'AMTG',
                  'L_caudalmiddlefrontal',
                  'L_IFG',
                  'L_inferiorparietal',
                  'L_lateralorbitofrontal',
                  'L_precuneus',
                  'L_superiorfrontal',
                  'ATL',
                  'L_middletemporal',
                  'R_IFG',
                  'R_lateralorbitofrontal',
                  'R_precuneus',
                  'R_ATL',
                  ]

damage_ratings = ['auditory', 'hand_arm']
for dataset, brain_data in data.items():
    for area, area_data in tqdm(brain_data.items()):
        if area not in selected_areas:
            continue
        print(area)
        for rat in damage_ratings:
            sorted_ws = sorted([(w, v[rat]) for w, v in ratings.items()], key=lambda item: item[1])
            for damage_amount in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25]:
                percent = int(len(sorted_ws)*damage_amount)
                lim_ctx_words = [w for w, val in sorted_ws[-percent:]]
                idxs = [ctx_words.index(w) for w in lim_ctx_words]
                for damage_type, damage_mtrx in mtrxs.items():
                    damaged_pmi_mtrx = numpy.copy(pmi_mtrx)
                    for idx in idxs:
                        damaged_pmi_mtrx[:, idx] = damage_mtrx[:, idx]
                        damaged_pmi_mtrx[idx, :] = damage_mtrx[idx, :]
                    vecs = {k : v for k, v in zip(ctx_words, damaged_pmi_mtrx)}
                    out_file = os.path.join(
                                    out, 
                                    'fernandino{}_{}_{}_{}_{}_{}_{}_{}.results'.format(
                                        dataset,
                                        area,
                                        corpus, 
                                        min_count, 
                                        win_size, 
                                        rat, 
                                        damage_type, 
                                        damage_amount)
                                    )
                    bins_rsa_test(words[dataset], ratings, vecs, area_data, out_file)
