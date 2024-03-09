import os
import re

from tqdm import tqdm

from qwikidata.linked_data_interface import get_entity_dict_from_api

def return_entity_path(title):
    ### output preparation
    re_ed_title = re.sub('\W', r'_', title)
    out_folder = re_ed_title[:3]
    out_file = os.path.join(out_folder, '{}.wiki_page'.format(re_ed_title))
    return out_file

for language in [
                 #'en', 
                 'it',
                 ]:
    if language == 'it':
        fol = 'wikipedia_09-01-2023_spacy_annotation_article_by_article_b'
    main_folder = os.path.join(
                               '..',
                               '..',
                               'dataset',
                               'corpora',
                               language,
                               fol
                               )
    aliases = dict()
    for experiment_id in ['one', 'two']:

        ### reading entity list
        entity_list_file = os.path.join(
                                        'entities',
                                        'wikidata_ids_{}.txt'.format(experiment_id)
                                        )
        assert os.path.exists(entity_list_file)
        with open(entity_list_file) as i:
            ent_lines = [l.strip().split('\t') for l in i.readlines()]
        for l_i, l in enumerate(ent_lines):
            if l_i == 0:
                print(l)
                assert l[0] == 'entity'
                assert l[1] == 'wikidata_id'
            assert len(l) == 2
        entities = {l[0] : l[1] for l in ent_lines[1:] if l[0][0]!='#'}

        ### checking wiki files
        for k, wikidata_id in tqdm(entities.items()):
            ent_dict = get_entity_dict_from_api(wikidata_id)
            main_alias = ent_dict['labels'][language]['value']
            #print(main_alias)
            aliases[k] = [
                          k,
                          main_alias, 
                          main_alias.capitalize(), 
                          ' '.join([w.capitalize() for w in k.split()]),
                          ]
            ### all combinations of casings
            if k == 'Madonna':
                aliases[k].append('Madonna_(cantante)')
            ### last names
            #if len(k.split()) > 1:
            #    aliases[k].append(k.split()[-1])
            if language in ent_dict['aliases'].keys():
                for al in ent_dict['aliases'][language]:
                    if len(al) < 3:
                        continue
                    main_alias = al['value']
                    variants = [main_alias, main_alias.capitalize(), ' '.join([w.capitalize() for w in k.split()])]
                    aliases[k].extend(variants)

            aliases = {k : sorted(set(v), key=lambda item : len(item), reverse=True) for k, v in aliases.items()}

            ### relevant paths
            final_paths = {k : list() for k in aliases.keys()}
            for k, k_aliases in aliases.items():
                for main_alias in k_aliases:
                    if len(main_alias) > 2:
                        file_k = return_entity_path(main_alias)
                        try:
                            assert os.path.exists(os.path.join(main_folder, file_k))
                        except AssertionError:
                            continue
                        final_paths[k].append(os.path.join(main_folder, file_k))
            import pdb; pdb.set_trace()
