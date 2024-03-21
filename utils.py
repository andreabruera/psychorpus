import numpy
import os
import pickle
import re

from tqdm import tqdm

def read_ratings(hand=False):
    ### sensory ratings
    file_path = os.path.join(
                             'data',
                             'Lancaster_sensorimotor_norms_for_39707_words.tsv',
                             )
    assert os.path.exists(file_path)
    relevant_keys = [
                     'Auditory.mean',
                     'Gustatory.mean',
                     'Haptic.mean',
                     'Olfactory.mean',
                     'Visual.mean',
                     #'Foot_leg.mean',
                     #'Hand_arm.mean', 
                     #'Head.mean', 
                     #'Mouth.mean', 
                     #'Torso.mean'
                     ]
    if hand:
        relevant_keys.append('Hand_arm.mean')
    norms = {k.lower().split('.')[0] : dict() for k in relevant_keys}
    with open(file_path) as i:
        counter = 0
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                header = line.copy()
                continue
            assert len(line) == len(header)
            marker = False
            for k in relevant_keys:
                try:
                    assert float(line[header.index(k)]) <= 5 
                except AssertionError:
                    #logging.info(line[0])
                    marker = True
            if marker:
                continue
            if len(line[0].split()) == 1:
                for k in relevant_keys:
                    val = float(line[header.index(k)])
                    ### minimum is 0, max is 5
                    assert val >= 0. and val <= 5.
                    curr_val = float(val) / 5
                    norms[k.lower().split('.')[0]][line[0].lower().strip()] = curr_val
    ### concreteness
    norms['concreteness'] = dict()
    with open(os.path.join('data', 'Concreteness_ratings_Brysbaert_et_al_BRM.txt')) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.strip().split('\t')
            ### minimum is 1, max is 5
            assert val >= 1. and val <= 5.
            curr_val = (float(line[2]) - 1) / (5 - 1)
            w = line[0].lower().strip()
            if w in norms['visual'].keys():
                norms['concreteness'][w] = curr_val
    ### checking that all went good...
    for k, v in norms.items():
        for w in v.keys():
            for k_two, v_two in norms.items():
                assert w in v_two.keys()
    ### putting the dictionary together
    final_norms = {k : {k_two : v_two[k] for k_two, v_two in norms.items()} for k in norms['concreteness'].keys()}

    return final_norms

def read_fernandino(vocab, pos, return_dict=False):

    words = {1 : list(), 2 : list()}
    subjects_data = {1 : dict(), 2 : dict()}
    full_subjects_data = {1 : dict(), 2 : dict()}
    pkl_path = os.path.join('data', 'fernandino_rsa.pkl')
    full_pkl_path = os.path.join('data', 'fernandino_pairwise.pkl')
    marker = False
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as i:
            subjects_data = pickle.load(i)
        with open(full_pkl_path, 'rb') as i:
            full_subjects_data = pickle.load(i)
        marker = True

    for d in words.keys():
        missing_idxs = list()
        ### words
        with open(os.path.join('data', 'fernandino{}_words.txt'.format(d))) as i:
            for l_i, l in enumerate(i):
                line = l.strip()
                if line != '':
                    if vocab[line] == 0:
                        missing_idxs.append(l_i)
                        print('missing: {}'.format([line, pos[line]]))
                        continue
                    words[d].append(line)
        ### similarities
        ### other anterior-frontal areas
        ### reading mapper
        if marker:
            continue

        mapper = dict()
        with open(os.path.join('data', 'colortable_desikan_killiany.txt')) as i:
            for l in i:
                l = re.sub('\s+', r'\t', l)
                line = l.strip().split('\t')
                assert len(line) > 2
                mapper[line[0]] = 'L_{}'.format(line[1])
                mapper[str(int(line[0])+35)] = 'R_{}'.format(line[1])
        folder = 'Study{}_neural_vectors_RSMs'.format(d)
        for brain_area_folder in tqdm(os.listdir(os.path.join('data', folder))):
            brain_area = re.sub(r'ALE|DK_|roi|_mask', '', brain_area_folder)
            if brain_area in mapper.keys():
                brain_area = mapper[brain_area]
            #print(brain_area)
            for f in os.listdir(os.path.join('data', folder, brain_area_folder,)):
                if 'txt' not in f:
                    continue
                mtrx = list()
                sub = f.split('_')[-1].replace('.txt', '')
                with open(os.path.join('data', folder, brain_area_folder, f)) as i:
                    for l_i, l in enumerate(i):
                        if l_i in missing_idxs:
                            continue
                        line = [sim for sim_i, sim in enumerate(l.strip().split('\t')) if sim_i not in missing_idxs]
                        mtrx.append(line)
                ### checks
                assert len(mtrx) == len(words[d])
                for line in mtrx:
                    assert len(line) == len(words[d])
                ### adding data
                if brain_area not in subjects_data[d].keys():
                    subjects_data[d][brain_area] = dict()
                    if return_dict:
                        full_subjects_data[d][brain_area] = dict()
                ### RSA
                ### removing diagonal
                subjects_data[d][brain_area][sub] = numpy.array([val for line_i, line in enumerate(mtrx) for val_i, val in enumerate(line) if val_i>line_i], dtype=numpy.float64).tolist()
                if return_dict:
                    full_subjects_data[d][brain_area][sub] = dict()
                    for w_one_i, w_one in enumerate(words[d]):
                        for w_two_i, w_two in enumerate(words[d]):
                            if w_two_i > w_one_i:
                                full_subjects_data[d][brain_area][sub][tuple(sorted([w_one, w_two]))] = float(mtrx[w_one_i][w_two_i])
    if not marker:
        with open(pkl_path, 'wb') as i:
            pickle.dump(subjects_data, i)
        with open(full_pkl_path, 'wb') as i:
            pickle.dump(full_subjects_data, i)

    if return_dict:
        return words, subjects_data, full_subjects_data
    else:
        return words, subjects_data

def read_men():
    sims = dict()
    with open(os.path.join('data', 'MEN', 'MEN_dataset_natural_form_full')) as i:
        for l in i:
            ### UK spelling correction...
            if 'donut' in l:
                l = l.replace('donut', 'doughnut')
            if 'colorful' in l:
                l = l.replace('colorful', 'colourful')
            line = l.strip().split()
            sims[(line[0], line[1])] = float(line[2])
    return sims

def read_simlex():
    sims = dict()
    with open(os.path.join('data', 'SimLex-999', 'SimLex-999.txt')) as i:
        for l_i, l in enumerate(i):
            if l_i==0:
                continue
            line = l.strip().split()
            sims[(line[0], line[1])] = float(line[3])
    return sims

def read_men_test():
    sims = dict()
    with open(os.path.join('data', 'MEN', 'MEN_dataset_lemma_form.test')) as i:
        for l in i:
            ### UK spelling correction...
            if 'donut' in l:
                l = l.replace('donut', 'doughnut')
            if 'colorful' in l:
                l = l.replace('colorful', 'colourful')
            line = l.strip().split()
            sims[(line[0].split('-')[0], line[1].split('-')[0])] = float(line[2])
    return sims

def build_ppmi_vecs(coocs, vocab, row_words, col_words):
    pmi_mtrx = numpy.array(
                             [
                              [coocs[vocab[w]][vocab[w_two]] if vocab[w_two] in coocs[vocab[w]].keys() else 0 for w_two in col_words]
                              for w in row_words])
    assert pmi_mtrx.shape[0] == len(row_words)
    assert pmi_mtrx.shape[1] == len(col_words)
    #matrix_[matrix_ != 0] = np.array(1.0/matrix_[matrix_ != 0])
    axis_one_sum = pmi_mtrx.sum(axis=1)
    axis_one_mtrx = numpy.divide(1, axis_one_sum, where=axis_one_sum!=0).reshape(-1, 1)
    ### raising to 0.75 as suggested in Levy & Goldberg 2015
    axis_zero_sum = numpy.power(pmi_mtrx, 0.75).sum(axis=0)
    axis_zero_sum = pmi_mtrx.sum(axis=0)
    axis_zero_mtrx = numpy.divide(1, axis_zero_sum, where=axis_zero_sum!=0).reshape(1, -1)
    total_sum = pmi_mtrx.sum()
    #trans_pmi_mtrx = numpy.multiply(numpy.multiply(numpy.multiply(pmi_mtrx,1/pmi_mtrx.sum(axis=1).reshape(-1, 1)), 1/pmi_mtrx.sum(axis=0).reshape(1, -1)), pmi_mtrx.sum())
    trans_pmi_mtrx = numpy.multiply(
                                    numpy.multiply(
                                                   numpy.multiply(
                                                                  pmi_mtrx,axis_one_mtrx), 
                                                   axis_zero_mtrx), 
                                    total_sum)
    trans_pmi_mtrx[trans_pmi_mtrx<1.] = 1
    trans_pmi_vecs = {w : numpy.log2(trans_pmi_mtrx[w_i]) for w_i, w in enumerate(row_words)}

    return trans_pmi_vecs
