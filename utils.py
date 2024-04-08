import numpy
import os
import pickle
import re

from tqdm import tqdm

def divide_binder_ratings(ratings):
    subdivisions = dict()
    with open(os.path.join('data', 'binder_sections.tsv')) as i:
        for l in i:
            line = l.strip().split('\t')
            if line[1] not in ratings['actor'].keys():
                print(line)
                continue
            if line[0] not in subdivisions.keys():
                subdivisions[line[0]] = list()
            subdivisions[line[0]].append(line[1])
    section_vecs = {w : {'{}_section'.format(k) : [ratings[w][dim] for dim in v] for k, v in subdivisions.items()} for w in ratings.keys()}
    return section_vecs

def read_exp48(words):
    vecs = dict()
    ### sensory ratings
    file_path = os.path.join(
                             'data',
                             'fernandino_experiential_ratings.tsv',
                             )
    assert os.path.exists(file_path)
    with open(file_path) as i:
        for l_i, l in enumerate(i):
            line = l.replace(',', '.').strip().split('\t')
            if l_i == 0:
                dimensions = [w.strip() for w in line[1:]]
                continue
            vecs[line[0].lower().strip()] = numpy.array(line[1:], dtype=numpy.float64)

    return vecs, list(dimensions)

def read_fernandino_ratings():
    ### sensory ratings
    file_path = os.path.join(
                             'data',
                             'fernandino_experiential_ratings.tsv',
                             )
    assert os.path.exists(file_path)
    norms = dict()
    with open(file_path) as i:
        counter = 0
        for l_i, l in enumerate(i):
            line = l.replace(',', '.').strip().split('\t')
            if l_i == 0:
                header = line.copy()
                continue
            assert len(line) == len(header)
            if len(line[0].split()) == 1:
                for h_i, h in zip(range(len(header)), header):
                    if h_i == 0:
                        continue
                    val = float(line[h_i])
                    ### minimum is 0, max is 5
                    assert val >= 0. and val <= 6.
                    curr_val = float(val) / 6.
                    if h not in norms.keys():
                        norms[h] = dict()
                    norms[h][line[0].lower().strip()] = curr_val
    ### checking that all went good...
    for k, v in norms.items():
        for w in v.keys():
            for k_two, v_two in norms.items():
                assert w in v_two.keys()
    ### putting the dictionary together
    final_norms = {k : {k_two : v_two[k] for k_two, v_two in norms.items()} for k in norms['Audition'].keys()}

    return final_norms

def read_binder_ratings():
    ### sensory ratings
    file_path = os.path.join(
                             'data',
                             'binder_ratings.tsv',
                             )
    assert os.path.exists(file_path)
    norms = dict()
    with open(file_path) as i:
        counter = 0
        for l_i, l in enumerate(i):
            line = l.replace(',', '.').strip().split('\t')
            if l_i == 0:
                header = line.copy()
                continue
            assert len(line) == len(header)
            for h_i, h in zip(range(len(header)), header):
                if h_i in [0, 1, 2, 3, 4] or h_i>69:
                    print(h)
                    continue
                if line[h_i] == 'na':
                    curr_val = numpy.nan
                else:
                    val = float(line[h_i])
                    ### minimum is 0, max is 5
                    assert val >= 0. and val <= 6.
                    curr_val = float(val) / 6.
                if h not in norms.keys():
                    norms[h] = dict()
                norms[h][line[1].lower().strip()] = curr_val
    assert len(norms.keys()) == 65
    ### checking that all went good...
    for k, v in norms.items():
        for w in v.keys():
            for k_two, v_two in norms.items():
                assert w in v_two.keys()
    ### putting the dictionary together
    final_norms = {k : {k_two : v_two[k] for k_two, v_two in norms.items()} for k in norms['Audition'].keys()}

    return final_norms

def read_ratings(hand=False, concreteness=False):
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
    if concreteness:
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
    final_norms = {k : {k_two : v_two[k] for k_two, v_two in norms.items()} for k in norms['visual'].keys()}

    return final_norms

def read_fernandino(vocab, pos, return_dict=False, avg_subjects=False):

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
                    try:
                        if vocab[line] == 0:
                            missing_idxs.append(l_i)
                            print('missing: {}'.format([line, pos[line]]))
                            continue
                    except KeyError:
                        print('missing: {} - unknown POS'.format(line))

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
            brain_area = re.sub(r'ALE_|DK_|roi|_mask', '', brain_area_folder)
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
    ### replicating results by fernandino
    if avg_subjects:
        for d in subjects_data.keys():
            for a in subjects_data[d].keys():
                avg_corrs = numpy.average([v for v in subjects_data[d][a].values()], axis=0)
                subjects_data[d][a] = {'all' : avg_corrs}
                if return_dict:
                    avg_sims = dict()
                    for _, tups in full_subjects_data[d][a].items():
                        for t, val in tups.items():
                            try:
                                avg_sims[t] = (avg_sims[t] + val)/2
                            except KeyError:
                                avg_sims[t] = val
                    full_subjects_data[d][a] = {'all' : avg_sims}

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

def build_ppmi_vecs(coocs, vocab, row_words, col_words, smoothing=False, power=1.):
    pmi_mtrx = numpy.array(
                             [
                              [coocs[vocab[w]][vocab[w_two]] if vocab[w_two] in coocs[vocab[w]].keys() else 0 for w_two in col_words]
                              for w in row_words])
    assert pmi_mtrx.shape[0] == len(row_words)
    assert pmi_mtrx.shape[1] == len(col_words)
    if power != 1.:
        pmi_mtrx = numpy.power(pmi_mtrx, power)
    #matrix_[matrix_ != 0] = np.array(1.0/matrix_[matrix_ != 0])
    axis_one_sum = pmi_mtrx.sum(axis=1)
    axis_one_mtrx = numpy.divide(1, axis_one_sum, where=axis_one_sum!=0).reshape(-1, 1)
    axis_zero_sum = pmi_mtrx.sum(axis=0)
    axis_zero_mtrx = numpy.divide(1, axis_zero_sum, where=axis_zero_sum!=0).reshape(1, -1)
    ### raising to 0.75 as suggested in Levy & Goldberg 2015
    if smoothing:
        total_sum = numpy.power(pmi_mtrx, 0.75).sum()
    else:
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

def read_brys_ratings():
    ### valence, arousal and dominance
    ratings = {
               'valence' : dict(),
               'arousal' : dict(),
               'dominance' : dict(),
               'concreteness' : dict(),
               'aoa' : dict(),
               }

    counter = 0
    with open(os.path.join('data', 'BRM-emot-submit.csv')) as i:
        for l in i:
            if counter == 0:
                counter += 1
                continue
            line = l.strip().split(',')
            ratings['valence'][line[1]] = (float(line[2]) - 1) / (9 - 1)
            ratings['arousal'][line[1]] = (float(line[5]) - 1) / (9 -1)
            ratings['dominance'][line[1]] = (float(line[8]) - 1) / (9 - 1)

    counter = 0
    with open(os.path.join('data', 'Concreteness_ratings_Brysbaert_et_al_BRM.txt')) as i:
        for l in i:
            if counter == 0:
                counter += 1
                continue
            line = l.strip().split('\t')
            curr_val = (float(line[2]) - 1) / (5 - 1)
            ratings['concreteness'][line[0]] = curr_val

    ### reading dataset aoa
    with open(os.path.join('data', 'kuperman_aoa.tsv')) as i:
        counter = 0
        for l in i:
            line = l.strip().replace(',', '.').split('\t')
            if counter == 0:
                header = [w.strip() for w in line]
                counter += 1
                continue
            word = line[0].lower()
            idx = header.index('Rating.Mean')
            if line[idx] == 'NA':
                print(word)
                continue
            ratings['aoa'][word] = float(line[idx])

    return ratings

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = numpy.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])
