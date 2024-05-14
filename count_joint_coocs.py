import os
import pickle

from tqdm import tqdm

corpora = [
           'wac',
           'opensubs',
           #'tagged_wiki',
           ]

selected_pos = ['PROPN', 'VERB', 'NOUN', 'ADV', 'ADJ', ]

for win_size in [
                 4, 
                 #20,
                 ]:
    for lang in [
                 #'de',
                 #'en',
                 'it',
                 ]:
        ### we load all models with min_count==10
        min_count = 10
        ### dealing with the basics: freqs and pos
        general_freqs = dict()
        general_pos = dict()
        for corpus in corpora:
            with open(os.path.join(
                                   'pickles', lang, corpus, 
                                   '{}_{}_uncased_word_freqs.pkl'.format(lang, corpus),
                                   ), 'rb') as i:
                freqs = pickle.load(i)
            print('updating frequencies...')
            with tqdm() as counter:
                for k, v in freqs.items():
                    try:
                        general_freqs[k] += v
                        counter.update(1)
                    except KeyError:
                        general_freqs[k] = v
            del freqs
            print('updating pos...')
            with open(os.path.join(
                                   'pickles', lang, corpus, 
                                   '{}_{}_uncased_word_pos.pkl'.format(lang, corpus),
                                   ), 'rb') as i:
                pos = pickle.load(i)
            with tqdm() as counter:
                for k, all_vs in pos.items():
                    for p, v in all_vs.items():
                        try:
                            general_pos[k][p] += v
                            counter.update(1)
                        except KeyError:
                            try:
                                general_pos[k][p] = v
                                counter.update(1)
                            except KeyError:
                                general_pos[k] = dict()
                                general_pos[k][p] = v
                                counter.update(1)
            del pos
        ### dumping pos and freqs...
        pkls = os.path.join('pickles', lang, 'joint')
        os.makedirs(pkls, exist_ok=True)
        freqs_file = os.path.join(pkls, '{}_joint_uncased_word_freqs.pkl'.format(lang))
        with open(freqs_file, 'wb') as o:
            pickle.dump(general_freqs, o)
        pos_file = os.path.join(pkls, '{}_joint_uncased_word_pos.pkl'.format(lang))
        with open(pos_file, 'wb') as o:
            pickle.dump(general_pos, o)

        if lang in ['de', 'it']:
            min_count = 10
        else:
            min_count = 100
        ### now taking care of the vocab
        general_vocab = dict()
        counter = 1
        print('creating the vocab...')
        with tqdm() as tqdm_count:
            for k, v in general_freqs.items():
                ### setting the minimum frequency
                if v > min_count:
                    ### checking pos
                    try:
                        dom_pos = sorted(list(general_pos[k].items()), key=lambda item : item[1], reverse=True)[0][0]
                        if dom_pos in selected_pos:
                            general_vocab[k] = counter
                            counter += 1
                            tqdm_count.update(1)
                        else:
                            general_vocab[k] = 0
                    except KeyError:
                        general_vocab[k] = 0
                else:
                    general_vocab[k] = 0
        del general_pos
        del general_freqs
        vocab_file = os.path.join(pkls, '{}_joint_uncased_vocab_min_{}.pkl'.format(lang, min_count))
        with open(vocab_file, 'wb') as o:
            pickle.dump(general_vocab, o)
        ### now finally joining coocs
        general_coocs = dict()
        ### loading min_count==10
        min_count = 10
        for corpus in corpora:
            try:
                with open(os.path.join(
                                       'pickles', lang, corpus, 
                                       '{}_{}_coocs_uncased_min_{}_win_{}.pkl'.format(lang, corpus, min_count, win_size),
                                       ), 'rb') as i:
                    coocs = pickle.load(i)
            except FileNotFoundError:
                print('this model is missing!')
                continue
            with open(os.path.join(
                                   'pickles', lang, corpus, 
                                   '{}_{}_uncased_vocab_min_{}.pkl'.format(lang, corpus, min_count),
                                   ), 'rb') as i:
                vocab = pickle.load(i)
                inv_vocab = {v : k for k, v in vocab.items()}
            print('counting the coocs...')
            with tqdm() as counter:
                for k_one, k_one_d in coocs.items():
                    if k_one == 0:
                        continue
                    w_one = inv_vocab[k_one]
                    try:
                        if general_vocab[w_one] == 0:
                            continue
                    except KeyError:
                        continue
                    for k_two, val in k_one_d.items():
                        if k_two == 0:
                            continue
                        w_two = inv_vocab[k_two]
                        try:
                            if general_vocab[w_two] == 0:
                                continue
                        except KeyError:
                            continue
                        try:
                            general_coocs[general_vocab[w_one]][general_vocab[w_two]] += val
                        except KeyError:
                            try:
                                general_coocs[general_vocab[w_one]][general_vocab[w_two]] = val
                            except KeyError:
                                general_coocs[general_vocab[w_one]] = dict()
                                general_coocs[general_vocab[w_one]][general_vocab[w_two]] = val
                        counter.update(1)
            del coocs

        if lang in ['de', 'it']:
            min_count = 10
        else:
            min_count = 100
        ### finally writing to file...
        coocs_file = os.path.join(pkls, '{}_joint_coocs_uncased_min_{}_win_{}.pkl'.format(lang, min_count, win_size))
        with open(coocs_file, 'wb') as o:
            pickle.dump(general_coocs, o)
        del general_coocs
