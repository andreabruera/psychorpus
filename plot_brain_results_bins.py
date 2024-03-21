import matplotlib
import numpy
import os
import pdb
import scipy

from matplotlib import pyplot
from scipy import stats
from tqdm import tqdm

results_folder = os.path.join('results', 'rsa', 'bins')
assert os.path.exists(results_folder)


### setting variables
files = {f.replace('_semantic_network', 'semantic-network').replace('_L_', '_L-').replace('_R_', '_R-').replace('hand_arm', 'hand') : f for f in os.listdir(results_folder)}
areas = set([f.split('_')[1] for f in files.keys()])
datasets = [1, 2]
damages = set([' '.join(f.split('_')[6:]).replace('.results', '') for f in files.keys()])
### grouping damages
damages_dict = dict()
for d in damages:
    if len(d.split()) == 2:
        try:
            damages_dict[d.split()[0]].append(d)
        except KeyError:
            damages_dict[d.split()[0]] = [d]
    if len(d.split()) == 3:
        try:
            damages_dict[' '.join(d.split()[:2])].append(d)
        except KeyError:
            damages_dict[' '.join(d.split()[:2])] = [d]
    if len(d.split()) == 4:
        try:
            damages_dict[' '.join(d.split()[:3])].append(d)
        except KeyError:
            damages_dict[' '.join(d.split()[:3])] = [d]

variables = set([f.split('_')[5] for f in files.keys()])

with tqdm() as counter:
    for variable in variables:
        for dataset in datasets:
            plot_folder = os.path.join('brain_plots', 'rsa', 'bins', str(dataset))
            os.makedirs(plot_folder, exist_ok=True)
            for area in areas:
                for damage, damage_variants in damages_dict.items():
                    if damage == 'undamaged':
                        continue
                    rel_files = list()
                    ###relevant files
                    for corr_file, original_file in files.items():
                        ### dataset
                        if 'fernandino{}'.format(dataset) in corr_file:
                            pass
                        else:
                            continue
                        ### area
                        if corr_file.split('_')[1] == area:
                            pass
                        else:
                            continue
                        ### damage
                        curr_damage = ' '.join(corr_file.split('_')[6:]).replace('.results', '')
                        if curr_damage == 'undamaged':
                            undamaged = os.path.join(results_folder, original_file)
                            continue
                        elif curr_damage in damage_variants:
                            pass
                        else:
                            continue
                        ### variable
                        if corr_file.split('_')[5] == variable:
                            pass
                        else:
                            continue
                        ### adding...
                        rel_files.append((curr_damage, os.path.join(results_folder, original_file)))
                    try:
                        assert len(rel_files) >= 1
                    except AssertionError:
                        continue
                    #print(len(rel_files))
                    assert results_folder in undamaged
                    dam_cases = list()
                    dam_results = list()
                    for f_damage, f in rel_files:
                        with open(f) as i:
                            for l_i, l in enumerate(i):
                                if l_i == 0:
                                    continue
                                line = l.strip().split('\t')
                                curr_var =  line[0].split('_')[0]
                                if curr_var == variable:
                                #    print(curr_var)
                                    print([curr_var, variable])
                                    dam_cases.append(f_damage)
                                    if 'nan' in line[1:]:
                                        dam_results.append(numpy.array([numpy.nan for _ in line[1:]]))
                                    else:
                                        dam_results.append(numpy.array(line[1:], dtype=numpy.float32))
                    with open(undamaged) as i:
                        for l_i, l in enumerate(i):
                            if l_i == 0:
                                continue
                            line = l.strip().split('\t')
                            curr_var =  line[0].split('_')[0]
                            #print([curr_var, variable])
                            if curr_var == variable:
                                #print(curr_var)
                                dam_cases.append('undamaged')
                                if 'nan' in line[1:]:
                                    dam_results.append(numpy.array([numpy.nan for _ in line[1:]]))
                                else:
                                    dam_results.append(numpy.array(line[1:], dtype=numpy.float32))
                    all_dam = {k : list() for k in set(dam_cases)}
                    for case, res in zip(dam_cases, dam_results):
                        all_dam[case].append(res)
                    if len(all_dam.keys()) == 0:
                        continue
                    #for k,v in all_dam.items():
                    #    print(numpy.array(v).shape)
                    ### plotting
                    fig, ax = pyplot.subplots(constrained_layout=True)
                    xs = list(range(5))
                    for case, dam_results in all_dam.items():
                        mod_dam_ys = numpy.nanmean(dam_results, axis=1)
                        if case == 'undamaged':
                            ax.plot(xs, mod_dam_ys, color='black', ls='-.', label=case.replace('_', ' '))
                        else:
                            ax.plot(xs, mod_dam_ys, label=case.replace('_', ' '))
                    ax.set_xlim(left=-.5, right=4.5)
                    ax.set_ylim(bottom=-0.02, top=0.1)
                    ax.set_xticks(ticks=range(5), labels=
                            ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1'])
                    ax.hlines(
                              y=[0.0],
                              xmin=-.2,
                              xmax=4.2,
                              alpha=0.3,
                              linestyles='-',
                              color='grey',
                              )
                    ax.hlines(
                              y=[-0.02, 0.02, 0.04, 0.06, 0.08, ],
                              xmin=-.2,
                              xmax=4.2,
                              alpha=0.1,
                              linestyles='--',
                              )
                    ax.legend()
                    file_out = os.path.join(
                                    plot_folder, 
                                    'fernandino{}_{}_{}_{}.jpeg'.format(dataset, variable, area, damage.replace(' ', '_'))
                                    )
                    print(file_out)
                    pyplot.savefig(file_out)
                    pyplot.clf()
                    pyplot.close()
                    counter.update(1)
