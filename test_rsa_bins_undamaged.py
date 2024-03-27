import numpy
import os
import pickle
import random
import scipy
import tqdm

from tqdm import tqdm

from utils import build_ppmi_vecs, read_fernandino_ratings, read_ratings, read_men, read_men_test, read_simlex, read_fernandino

def bins_rsa_test(words, ratings, model, brain_data, splits, out_file):

    n_items = 8
    dataset_results = dict()
    relevant_keys = ratings[words[0]].keys()

    bins = {'{}_{}'.format(case, i) : list() for i in range(len(splits)) for case in relevant_keys}

    for case_i, case in enumerate(relevant_keys):
        print(case)
        counter = 0
        for beg, end in tqdm(splits):
            #bin_words = [w for w in words if ratings[w][case]>=beg and ratings[w][case]<=end]
            bin_words = [w for w in words if w in ratings.keys() and ratings[w][case]>=beg and ratings[w][case]<=end]
            #print(len(bin_words))
            if len(bin_words) < n_items:
                print('nan')
                bins['{}_{}'.format(case, counter)] = [numpy.nan for s in brain_data.keys()]
                counter += 1
                continue
            bin_results = list()
            for _ in range(100):
                iter_results = list()
                current_bin_words = random.sample(bin_words, k=n_items)

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
            #print(len(v))
            assert len(v) == len(brain_data.keys())
            o.write('{}\t'.format(k))
            for val in v:
                o.write('{}\t'.format(val))
            o.write('\n')
fern_ratings = read_fernandino_ratings(hand=True)

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
                       'en_{}_coocs_uncased_min_{}_win_{}_pmi75_rows_cols.pkl'.format(corpus, min_count, win_size),
                       ), 'rb') as i:
    ctx_words = pickle.load(i)
mtrxs = dict()
with open(os.path.join(
                       'pickles', 'en', corpus, 
                       'en_{}_coocs_uncased_min_{}_win_{}_pmi.pkl'.format(corpus, min_count, win_size),
                       ), 'rb') as i:
    pmi_mtrx = pickle.load(i)


selected_areas = [
                  '_semantic_network',
                  #'AMTG',
                  #'L_caudalmiddlefrontal',
                  'L_IFG',
                  'L_inferiorparietal',
                  #'L_lateralorbitofrontal',
                  'L_precuneus',
                  #'L_superiorfrontal',
                  #'ATL',
                  #'L_middletemporal',
                  #'R_IFG',
                  #'R_lateralorbitofrontal',
                  #'R_precuneus',
                  #'R_ATL',
                  ]
unit = 25
assert 100 % unit == 0
splits = [(i*0.01, (i+unit)*0.01) for i in range(0, 100, unit)]
relevant_keys = fern_ratings['accordion'].keys()

for mode in [
            'averaged_sub_mtrxs', 
            'individual_sub_mtrxs',
            ]:
    for dataset, brain_data in data.items():
        for area, area_data in tqdm(brain_data.items()):
            if area not in selected_areas:
                continue
            out = os.path.join(
                               'results', 
                               'fernandino{}'.format(dataset), 
                               'bins',
                               mode,
                               area,
                               '75{}_{}_{}'.format(corpus, min_count, win_size),
                               'undamaged',
                               'undamaged',
                               )
            os.makedirs(out, exist_ok=True)
            print(out)
            for damage_amount in ['undamaged']:
                vecs = {k : v for k, v in zip(ctx_words, pmi_mtrx)}
                out_file = os.path.join(
                                out, 
                                '{}.results'.format(
                                        damage_amount), 
                                )
                bins_rsa_test(words[dataset], fern_ratings, vecs, area_data, splits, out_file)
