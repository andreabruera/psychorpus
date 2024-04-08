import multiprocessing
import numpy
import os
import pickle
import random
import scipy
import tqdm

from tqdm import tqdm

from utils import build_ppmi_vecs, read_fernandino_ratings, read_ratings, read_men, read_men_test, read_simlex, read_fernandino

def bins_rsa_test(inputs):
    mode = inputs[0]
    dataset = inputs[1]
    area = inputs[2]
    variable = inputs[3]
    out_file = inputs[4]

    words = all_words[mode][dataset]
    brain_data = all_data[mode][dataset][area]
    #import pdb; pdb.set_trace()
    model = variables[variable]

    dataset_results = dict()

    bins = {'{}_{}'.format(case, i) : list() for i in range(len(splits)) for case in relevant_keys}

    for case_i, case in enumerate(relevant_keys):
        #print(case)
        counter = 0
        for beg, end in tqdm(splits):
            bin_words = [w for w in words if w in ratings.keys() and ratings[w][case]>=beg and ratings[w][case]<=end]
            if len(bin_words) < n_items:
                print('nan')
                bins['{}_{}'.format(case, counter)] = [numpy.nan for s in brain_data.keys()]
                counter += 1
                continue
            bin_results = list()
            random.seed(seed)
            for _ in range(100):
                iter_results = list()
                current_bin_words = random.sample(bin_words, k=n_items)
                if type(model[list(model.keys())[0]]) == float:
                    sim_model = [-abs(model[w_one]-model[w_two]) for w_one_i, w_one in enumerate(current_bin_words) for w_two_i, w_two in enumerate(current_bin_words) if w_two_i>w_one_i]
                else:
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

    print(out_file)
    with open(out_file, 'w') as o:
        o.write('bin\tresults\n')
        for k, v in bins.items():
            assert len(v) == len(brain_data.keys())
            #print(len(v))
            o.write('{}\t'.format(k))
            for val in v:
                o.write('{}\t'.format(val))
            o.write('\n')

debugging = False
#debugging = True
global n_items
n_items = 20

selected_areas = [
                  'semantic_network',
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
### loading undamage model
corpus = 'opensubs'
min_count = 10
win_size = 4
marker = 'pmi_smooth75'
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

global ratings
ratings = {k : v for k, v in read_fernandino_ratings().items() if k in ctx_words}
unit = 25
assert 100 % unit == 0
global splits
splits = [(i*0.01, (i+unit)*0.01) for i in range(0, 100, unit)]

seed = 12
global relevant_keys
relevant_keys = [
            'Audition',
            'Practice',
            'UpperLimb',
            'Touch',
            'Vision',
            'Taste',
            'Smell',
            'Sound',
            ]
global variables
variables = {k : {w : vals[k] for w, vals in ratings.items()} for k in relevant_keys}
variables['{}_{}'.format(marker, full_corpus)] = {w : mtrx[w_i] for w_i, w in enumerate(ctx_words)}
### dummy variables
global vocab
vocab = {k : 1 for k in ratings.keys()}
global pos
pos = {k : 1 for k in ratings.keys()}

print('now loading the models...')
all_words = dict()
all_data = dict()
for mode in [
            'averaged_sub_mtrxs', 
            'individual_sub_mtrxs',
            ]:
    if mode == 'averaged_sub_mtrxs':
        words, _, data = read_fernandino(vocab, pos, return_dict=True, avg_subjects=True)
    elif mode == 'individual_sub_mtrxs':
        words, _,data = read_fernandino(vocab, pos, return_dict=True, avg_subjects=False)
    all_words[mode] = words.copy()
    all_data[mode] = data.copy()
    del words
    del data
print('loaded!')
inputs = list()
print('now preparing inputs...')
for mode, data in all_data.items():
    for dataset, brain_data in data.items():
        for area, area_data in brain_data.items():
            if area not in selected_areas:
                continue
            for variable, vecs in variables.items():
                ### undamaged
                out = os.path.join(
                                   'results', 
                                   'fernandino{}'.format(dataset), 
                                   'bins',
                                   mode,
                                   'undamaged',
                                   area,
                                   variable,
                                   variable,
                                   'undamaged',
                                   'undamaged',
                                   )
                os.makedirs(out, exist_ok=True)
                #print(out)
                out_file = os.path.join(
                                out, 
                                'undamaged.results'
                                )
                inputs.append([mode, dataset, area, variable, out_file])
print('ready!')
if debugging:
    #bins_rsa_test(words[dataset], fern_ratings, vecs, area_data, splits, out_file)
    for inp in inputs:
        bins_rsa_test(inp)
else:
    with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
        pool.map(bins_rsa_test, inputs)
        pool.terminate()
        pool.join()
