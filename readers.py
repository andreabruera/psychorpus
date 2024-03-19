import os
import re

from tqdm import tqdm

def wiki_reader(args, folder_path, pos=False, file_paths=False):
    pos_mapper = load_pos_mappers(args)
    sentences = list()
    paths = list()
    ### in wiki the file is a folder...
    for file_path in os.listdir(folder_path):
        if pos:
            sentence = {
                        'word' : list(), 
                        'pos' : list(), 
                        }
        else:
            sentence = {
                        'word' : list(), 
                        }
        with open(os.path.join(folder_path, file_path)) as i:
            for l in i:
                #line = l.replace('[[[', '').replace(']]]', '')
                line = re.sub('\s+', r' ', l)
                line = line.split()
                sentence['word'].extend(line)
                if len(sentence['word']) > 1:
                    #yield(sentence)
                    sentences.append(sentence)
                    paths.append(file_path)
                '''
                ### grouping by chunks of 512 tokens
                sentence['word'].extend(line)
                if len(sentence) >= 512:
                    #yield sentence
                    sentences.append(sentence)
                    sentence = {
                        'word' : list(), 
                        }
                '''
                sentence = {
                    'word' : list(), 
                    }
            if len(sentence['word']) > 1:
                #yield(sentence)
                sentences.append(sentence)
                paths.append(file_path)
        if len(sentence['word']) > 1:
            #yield(sentence)
            sentences.append(sentence)
            paths.append(file_path)

    if file_paths:
        return sentences, paths
    else:
        return sentences

def wac_reader(args, file_path, pos=False):
    pos_mapper = load_pos_mappers(args)
    sentences = list()
    with open(file_path) as i:
        if pos:
            sentence = {
                        'word' : list(), 
                        'pos' : list(), 
                        }
        else:
            sentence = {
                        'word' : list(), 
                        }
        for l in i:
            line = l.strip().split('\t')
            if line[0][:4] == '</s>':
                #yield sentence
                sentences.append(sentence)
                if pos:
                    sentence = {
                                'word' : list(), 
                                'pos' : list(), 
                                }
                else:
                    sentence = {
                                'word' : list(), 
                                }
            elif line[0][0] == '<':
                continue
            if len(line) < 2:
                continue
            else:
                if '$' in line[1]:
                    continue
                else:
                    sentence['word'].append(line[0])
                    if pos:
                        try:
                            pos = pos_mapper[line[2]]
                        except KeyError:
                            pos = line[2]
                        sentence['pos'].append(pos)
        if len(sentence['word']) > 1:
            #yield(sentence)
            sentences.append(sentence)
    return sentences

def tagged_wiki_reader(args, folder_path, pos=False):
    pos_mapper = load_pos_mappers(args)
    sentences = list()
    ### in tagged wiki the file is a folder...
    for file_path in os.listdir(folder_path):
        if pos:
            sentence = {
                        'word' : list(), 
                        'pos' : list(), 
                        }
        else:
            sentence = {
                        'word' : list(), 
                        }
        with open(os.path.join(folder_path, file_path)) as i:
            for l in i:
                line = l.strip().split('\t')
                if line[0] == '<EOS>':
                    #yield sentence
                    sentences.append(sentence)
                    if pos:
                        sentence = {
                                    'word' : list(), 
                                    'pos' : list(), 
                                    }
                    else:
                        sentence = {
                                    'word' : list(), 
                                    }
                elif len(line) < 4:
                    continue
                else:
                    w = re.sub('\W+', '_', line[0])
                    if line[2] == 'ENT':
                        w = '{}_#ent#'.format(w)
                    if w != '_':
                        sentence['word'].append(w)
                        if pos:
                            if line[2] == 'ENT':
                                pos = 'ENT'
                            else:
                                try:
                                    pos = pos_mapper[line[3]]
                                except KeyError:
                                    pos = line[3]
                            sentence['pos'].append(pos)
            if len(sentence['word']) > 1:
                #yield(sentence)
                sentences.append(sentence)
    return sentences

def opensubs_reader(args, folder_path, pos=False):
    pos_mapper = load_pos_mappers(args)
    sentences = list()
    ### in opensubs the file is a folder...
    for file_path in folder_path:
        if pos:
            sentence = {
                        'word' : list(), 
                        'pos' : list(), 
                        }
        else:
            sentence = {
                        'word' : list(), 
                        }
        with open(os.path.join(file_path)) as i:
            for l in i:
                line = l.strip().split('\t')
                if line[0] == '<EOS>':
                    #yield sentence
                    sentences.append(sentence)
                    if pos:
                        sentence = {
                                    'word' : list(), 
                                    'pos' : list(), 
                                    }
                    else:
                        sentence = {
                                    'word' : list(), 
                                    }
                elif len(line) < 4:
                    continue
                else:
                    w = re.sub('\W+', '_', line[0])
                    if w != '_':
                        sentence['word'].append(w)
                        if pos:
                            try:
                                pos = pos_mapper[line[2]]
                            except KeyError:
                                pos = line[2]
                            sentence['pos'].append(pos)
            if len(sentence['word']) > 1:
                #yield(sentence)
                sentences.append(sentence)
    return sentences

def bnc_reader(args, file_path, pos=False):
    pos_mapper = load_pos_mappers(args)
    sentences = list()
    with open(file_path) as i:
        if pos:
            sentence = {
                        'word' : list(), 
                        'pos' : list(), 
                        }
        else:
            sentence = {
                        'word' : list(), 
                        }
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.strip().split('\t')
            if line[0] == '<EOS>':
                #yield sentence
                sentences.append(sentence)
                if pos:
                    sentence = {
                                'word' : list(), 
                                'pos' : list(), 
                                }
                else:
                    sentence = {
                                'word' : list(), 
                                }
            elif len(line) < 2:
                continue
            else:
                w = re.sub('\W+', '_', line[0])
                if w != '_':
                    sentence['word'].append(w)
                    if pos:
                        try:
                            pos = pos_mapper[line[1]]
                        except KeyError:
                            pos = line[1]
                        sentence['pos'].append(pos)
        if len(sentence['word']) > 1:
            #yield(sentence)
            sentences.append(sentence)
    return sentences

def paths_loader(args, pos=False):
    ### loading sentences
    print('now collecting paths...')
    basic_folder = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', 'corpora', args.language)
    assert os.path.exists(basic_folder)
    if args.corpus in ['wiki', 'tagged_wiki']:
        marker = args.corpus.replace('wiki', '')
        wiki_path = os.path.join(basic_folder, '{}_wiki_{}art_by_art_2024-01-01'.format(args.language, marker))
        assert os.path.exists(wiki_path)
        ### for wikipedia we do not take files but folders!
        paths = [os.path.join(wiki_path, direc) for direc in os.listdir(wiki_path)]
    if args.corpus == 'wac':
        if args.language == 'en':
            wac_path = os.path.join(basic_folder, 'PukWaC_smaller_files')
        if args.language == 'de':
            wac_path = os.path.join(basic_folder, 'sdewac-v3-tagged_smaller_files')
        if args.language == 'it':
            wac_path = os.path.join(basic_folder, 'itwac_smaller_files')
        assert os.path.exists(wac_path)
        paths = [os.path.join(root, f) for root, direc, filez in os.walk(wac_path) for f in filez]
    if args.corpus == 'opensubs':
        opensubs_path = os.path.join(basic_folder, 'opensubs-2018_parsed_{}'.format(args.language))
        assert os.path.exists(opensubs_path)
        ### for opensubs we split files in bunches of 1000 files... 
        paths = [os.path.join(root, f) for root, direc, filez in os.walk(opensubs_path) for f in filez]
        paths = [paths[start:start+1000] for start in range(0, len(paths), 1000)]
    if args.corpus == 'bnc':
        if args.language != 'en':
            raise RuntimeError('BNC is obviously only available in English!')
        bnc_path = os.path.join(basic_folder, 'BNC_tagged')
        assert os.path.exists(bnc_path)
        paths = [os.path.join(root, f) for root, direc, filez in os.walk(bnc_path) for f in filez]
    print('number of paths collected: {}'.format(len(paths)))
    return paths

def load_pos_mappers(args):
    if args.corpus == 'bnc':
        mapper = {
                  'SUBST' : 'NOUN',
                  'UNC' : 'UNK',
                  }
    if args.corpus == 'opensubs':
        mapper = {
                  'CCONJ' : 'CONJ',
                  'UNC' : 'UNK',
                  'ADP' : 'PREP',
                  }
    if args.corpus == 'wac':
        if args.language == 'it':
            mapper = {
                      'NPR' : 'PROPN',
                      'PRE' : 'PREP',
                      'PRO:demo' : 'PRON',
                      'PRO:indef' : 'PRON',
                      'PRO:pers' : 'PRON',
                      'PRO:poss' : 'PRON',
                      'ADV:mente' : 'ADV',
                      'AUX:fin' : 'AUX',
                      'AUX:fin:cli' : 'AUX',
                      'AUX:geru' : 'AUX',
                      'AUX:geru:cli' : 'AUX',
                      'AUX:infi' : 'AUX',
                      'AUX:infi:cli' : 'AUX', 
                      'AUX:ppast' : 'AUX',
                      'AUX:ppre' : 'AUX',
                      'DET:demo' : 'DET',
                      'DET:indef' : 'DET',
                      'DET:num' : 'DET',
                      'DET:poss' : 'DET',
                      'DET:wh' : 'DET',
                    }
        if args.language == 'en':
            mapper = {
                      'VHZ' : 'VERB',
                      'VV' : 'VERB',
                      'VVD' : 'VERB',
                      'VVG' : 'VERB',
                      'VVN' : 'VERB',
                      'VVP' : 'VERB',
                      'VBZ' : 'VERB',
                      'WP' : 'PRON',
                      'WP$' : 'PRON',
                      'PP' : 'PRON',
                      'PP$' : 'PRON',
                      'WRB' : 'ADV',
                      'CC' : 'CONJ',
                      'NN' : 'NOUN',
                      'NNS' : 'NOUN',
                      'NP' : 'PROPN',
                      'NPS' : 'PROPN',
                      'RB' : 'ADV',
                      'RBR' : 'ADV',
                      'RBS' : 'ADV',
                      'VB' : 'AUX',
                      'VBD' : 'AUX',
                      'VBG' : 'AUX',
                      'VBN' : 'AUX',
                      'VBP' : 'AUX',
                      'VBZ' : 'AUX',
                      'VH' : 'AUX',
                      'VHD' : 'AUX',
                      'VHG' : 'AUX',
                      'VHN' : 'AUX',
                      'VHP' : 'AUX',
                      'VHZ' : 'AUX',
                      'JJ' : 'ADJ',
                      'JJR' : 'ADJ',
                      'JJS' : 'ADJ',
                      }
        if args.language == 'de':
            mapper = {
                    }
    if 'wiki' in args.corpus:
        mapper = {
                  }
    return mapper

