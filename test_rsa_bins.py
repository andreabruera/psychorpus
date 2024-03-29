import numpy
import os
import pickle
import random
import scipy
import tqdm

from tqdm import tqdm

from utils import build_ppmi_vecs, read_fernandino_ratings, read_ratings, read_men, read_men_test, read_simlex, read_fernandino

def bins_rsa_test(words, ratings, model, brain_data, splits, out_file):

    n_items = 20
    dataset_results = dict()
    relevant_keys = ratings[words[0]].keys()

    bins = {'{}_{}'.format(case, i) : list() for i in range(len(splits)) for case in relevant_keys}

    for case_i, case in enumerate(relevant_keys):
        print(case)
        counter = 0
        for beg, end in tqdm(splits):
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
                sim_model = [1-scipy.spatial.distance.cosine(model[w_one], model[w_two]) for w_one_i, w_one in enumerate(current_bin_words) for w_two_i, w_two in enumerate(current_bin_words) if w_two_i>w_one_i]
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
ratings = read_ratings(hand=True)

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
full_corpus = '{}_coocs_uncased_min_{}_win_{}'.format(corpus, min_count, win_size)
marker = 'pmi_smooth75'
out_f = os.path.join(
                     'damaged_pickles', 'en', full_corpus, marker, 
                       )
assert os.path.exists(out_f)
with open(os.path.join(
                        out_f,
                       'words_rows_cols.pkl',
                       ), 'rb') as i:
    ctx_words = pickle.load(i)
with open(os.path.join(
                       out_f,
                       'undamaged_coocs.pkl',
                       ), 'rb') as i:
    mtrx = pickle.load(i)
assert mtrx.shape == (len(ctx_words), len(ctx_words))
powers = [
          #0.05, 
          #0.1,
          0.25, 0.5, 0.75,
          #1.25, 1.5, 1.75
          ]

mtrxs = dict()
#mtrxs['zeroing'] = dict()
#mtrxs['oneing'] = dict()

'''
for r_c in ['rand_cols', 'rand_rows']:
    with open(os.path.join(
                               out_f,
                               '{}.pkl'.format(r_c),
                           ), 'rb') as i:
        mtrxs[r_c] = pickle.load(i)
    ### powers
    ### power + shuffle
    for power in powers:
        with open(os.path.join(
                           out_f,
                               '{}_pow_{}.pkl'.format(r_c, power),
                               ), 'rb') as i:
            mtrxs['{}_pow_{}'.format(r_c, power)] = pickle.load(i)

'''
### power only
for power in powers:
    with open(os.path.join(
                           out_f,
                           'pow_{}.pkl'.format(power),
                           ), 'rb') as i:
        mtrxs['pow_{}'.format(power)] = pickle.load(i)
    mtrxs['{}ing'.format(power)] = dict()

selected_areas = [
                  '_semantic_network',
                  ###
                  #'L_IFG',
                  #'L_inferiorparietal',
                  #'L_precuneus',
                  ###
                  #'AMTG',
                  #'L_caudalmiddlefrontal',
                  #'L_lateralorbitofrontal',
                  #'L_superiorfrontal',
                  #'ATL',
                  #'L_middletemporal',
                  #'R_IFG',
                  #'R_lateralorbitofrontal',
                  #'R_precuneus',
                  #'R_ATL',
                  ]

fern_ratings = read_fernandino_ratings(hand=True)
unit = 25
assert 100 % unit == 0
splits = [(i*0.01, (i+unit)*0.01) for i in range(0, 100, unit)]
relevant_keys = fern_ratings['accordion'].keys()

damage_ratings = [
                  'auditory', 
                  'hand_arm',
                  ]
for mode in [
            'averaged_sub_mtrxs', 
            'individual_sub_mtrxs',
            ]:
    if mode == 'averaged_sub_mtrxs':
        words, _, data = read_fernandino(vocab, pos, return_dict=True, avg_subjects=True)
    elif mode == 'individual_sub_mtrxs':
        words, _,data = read_fernandino(vocab, pos, return_dict=True, avg_subjects=False)
    for dataset, brain_data in data.items():
        for area, area_data in tqdm(brain_data.items()):
            if area not in selected_areas:
                continue
            '''
            ### undamaged
            out = os.path.join(
                               'results', 
                               'fernandino{}'.format(dataset), 
                               'bins',
                               mode,
                               area,
                               '{}_{}_{}'.format(corpus, min_count, win_size),
                               marker,
                               'undamaged',
                               'undamaged',
                               )
            os.makedirs(out, exist_ok=True)
            print(out)
            for damage_amount in ['undamaged']:
                vecs = {k : v for k, v in zip(ctx_words, mtrx)}
                out_file = os.path.join(
                                out, 
                                '{}.results'.format(
                                        damage_amount), 
                                )
                bins_rsa_test(words[dataset], fern_ratings, vecs, area_data, splits, out_file)
            '''
            ### damaged
            for rat in damage_ratings:
                ### here we use sensorimotor ratings
                sorted_ws = sorted([(w, v[rat]) for w, v in ratings.items() if w in ctx_words], key=lambda item: item[1])
                for damage_amount in [
                                      0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]:
                    percent = int(len(sorted_ws)*damage_amount)
                    lim_ctx_words = [w for w, val in sorted_ws[-percent:]]
                    idxs = [ctx_words.index(w) for w in lim_ctx_words]
                    assert len(idxs) > 0

                    ### other damages
                    for damage_type, damage_mtrx in mtrxs.items():
                        print(damage_type)
                        damaged_pmi_mtrx = numpy.copy(mtrx)
                        if damage_type == 'zeroing':
                            for idx in idxs:
                                damaged_pmi_mtrx[:, idx][idxs] = 0.
                        elif damage_type == 'oneing':
                            for idx in idxs:
                                damaged_pmi_mtrx[:, idx][idxs] = 1.
                        elif 'ing' in damage_type:
                            for idx in idxs:
                                damaged_pmi_mtrx[:, idx][idxs] = mtrxs['pow_{}'.format(damage_type.replace('ing', ''))][:, idx][idxs] 
                        elif damage_type == 'rand_rows':
                            for idx in idxs:
                                damaged_pmi_mtrx[idx, :] = numpy.array(random.sample(damaged_pmi_mtrx[idx, :].tolist(), k=len(ctx_words)))
                        else:
                            for idx in idxs:
                                damaged_pmi_mtrx[:, idx] = damage_mtrx[:, idx]
                                #damaged_pmi_mtrx[idx, :] = damage_mtrx[idx, :]
                        vecs = {k : v for k, v in zip(ctx_words, damaged_pmi_mtrx)}
                        out = os.path.join(
                                           'results', 
                                           'fernandino{}'.format(dataset), 
                                           'bins',
                                           mode,
                                           area,
                                           '{}_{}_{}'.format(corpus, min_count, win_size),
                                           marker,
                                           rat,
                                           damage_type,
                                           )
                        os.makedirs(out, exist_ok=True)
                        print(out)
                        out_file = os.path.join(
                                        out, 
                                        '{}.results'.format(
                                            damage_amount)
                                        )
                        bins_rsa_test(words[dataset], fern_ratings, vecs, area_data, splits, out_file)
