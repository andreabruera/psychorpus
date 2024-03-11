import numpy
import os

def read_ratings():
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
