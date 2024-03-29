import matplotlib
import numpy
import os
import pdb
import scipy

from matplotlib import pyplot
from scipy import stats
from tqdm import tqdm

unit = 25
assert 100 % unit == 0
splits = [(i*0.01, (i+unit)*0.01) for i in range(0, 100, unit)]


with tqdm() as counter:
    for dataset in [1, 2]:
        results_folder = os.path.join(
                                      'results', 
                                      'fernandino{}'.format(dataset), 
                                      'bins',
                                      )
        assert os.path.exists(results_folder)
        results = dict()
        for root, direc, fz in os.walk(results_folder):
            for f in fz:
                split_root = root.split('/')
                damage_type = split_root[-1]
                damage_variable = split_root[-2]
                marker= split_root[-3]
                model = '{}_{}'.format(marker, split_root[-4])
                area = split_root[-5]
                mode = split_root[-6]
                if model not in results.keys():
                    results[model] = dict()
                if mode not in results[model].keys():
                    results[model][mode] = dict()
                if area not in results[model][mode].keys():
                    results[model][mode][area] = dict()
                if damage_variable not in results[model][mode][area].keys():
                    results[model][mode][area][damage_variable] = dict()
                if damage_type not in results[model][mode][area][damage_variable].keys():
                    results[model][mode][area][damage_variable][damage_type] = dict()
                with open(os.path.join(root, f)) as i:
                    for l_i, l in enumerate(i):
                        if l_i == 0:
                            continue
                        line = l.strip().split('\t')
                        curr_var =  line[0].split('_')[0]
                        case = f.replace('.results', '')
                        if 'nan' in line[1:]:
                            res = numpy.array([numpy.nan for _ in line[1:]])
                        else:
                            res = numpy.array(line[1:], dtype=numpy.float32)
                        if curr_var not in results[model][mode][area][damage_variable][damage_type].keys():
                            results[model][mode][area][damage_variable][damage_type][curr_var] = dict()
                        if case not in results[model][mode][area][damage_variable][damage_type][curr_var].keys():
                            results[model][mode][area][damage_variable][damage_type][curr_var][case] = list()
                        results[model][mode][area][damage_variable][damage_type][curr_var][case].append(res)

        for model, model_d in results.items():
            for mode, mode_d in model_d.items():
                if 'average'  in mode:
                    ymin=-15
                    ymax=27
                else:
                    ymin=-9
                    ymax=15
                hlines=[y*0.01 for y in range(ymin+1, ymax, 2)]
                for area, area_d in mode_d.items():
                    for damage_variable, d_var_d in area_d.items():
                        if damage_variable == 'undamaged':
                            continue
                        for damage_type, d_type_d in d_var_d.items():
                            plots_folder = os.path.join(
                                                        'plots', 
                                                        'fernandino{}'.format(dataset), 
                                                        'bins',
                                                        mode,
                                                        area, 
                                                        model, 
                                                        damage_variable,
                                                        damage_type,
                                                        )
                            os.makedirs(plots_folder, exist_ok=True)
                            for curr_var, c_var_d in d_type_d.items():

                                ### plotting
                                fig, ax = pyplot.subplots(constrained_layout=True)
                                xs = list(range(len(splits)))
                                for case, dam_results in c_var_d.items():
                                    #print(dam_results)
                                    mod_dam_ys = numpy.nanmean(dam_results, axis=1)
                                    ax.plot(xs, mod_dam_ys, label=case.replace('_', ' '))
                                ### undamaged
                                undam_results = results[model][mode][area]['undamaged']['undamaged'][curr_var]['undamaged']
                                mod_undam_ys = numpy.nanmean(undam_results, axis=1)
                                ax.plot(xs, mod_undam_ys, label='undamaged', color='black', linestyle='-')
                                ax.set_xlim(left=-.5, right=len(splits)-.5)
                                ax.set_ylim(bottom=ymin*0.01, top=ymax*0.01)
                                ax.set_xticks(ticks=xs, labels=
                                        ['{}-{}'.format(s[0], s[1]) for s in splits])
                                ax.hlines(
                                          y=[0.0],
                                          xmin=-.2,
                                          xmax=len(splits)-.8,
                                          alpha=0.3,
                                          linestyles='-',
                                          color='grey',
                                          )
                                ax.hlines(
                                          y=hlines,
                                          xmin=-.2,
                                          xmax=len(splits)-.8,
                                          alpha=0.1,
                                          linestyles='--',
                                          )
                                ax.legend()
                                file_out = os.path.join(
                                                plots_folder, 
                                                '{}.jpeg'.format(curr_var),
                                                )
                                print(file_out)
                                pyplot.savefig(file_out)
                                pyplot.clf()
                                pyplot.close()
                                counter.update(1)
