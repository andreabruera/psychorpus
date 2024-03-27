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
        results_folder = os.path.join('results', 'fernandino{}'.format(dataset), 'bins')
        assert os.path.exists(results_folder)
        for area in os.listdir(results_folder):
            plots_folder = os.path.join('plots', 'fernandino{}'.format(dataset), 'bins',area, 'opensubs_10_4', 'undamaged',)
            os.makedirs(plots_folder, exist_ok=True)
            dam_cases = list()
            dam_results = list()
            f = os.path.join(results_folder, area, 'opensubs_10_4', 'undamaged', 'undamaged.results')
            assert os.path.exists(f)
            with open(f) as i:
                for l_i, l in enumerate(i):
                    if l_i == 0:
                        continue
                    line = l.strip().split('\t')
                    curr_var =  line[0].split('_')[0]
                    dam_cases.append(curr_var)
                    if 'nan' in line[1:]:
                        dam_results.append(numpy.array([numpy.nan for _ in line[1:]]))
                    else:
                        dam_results.append(numpy.array(line[1:], dtype=numpy.float32))
            all_dam = {k : list() for k in set(dam_cases)}
            for case, res in zip(dam_cases, dam_results):
                all_dam[case].append(res)
            if len(all_dam.keys()) == 0:
                continue
            for k,v in all_dam.items():
                print(numpy.array(v).shape)
            ### plotting
            fig, ax = pyplot.subplots(constrained_layout=True)
            xs = list(range(len(splits)))
            for case, dam_results in all_dam.items():
                print(dam_results)
                mod_dam_ys = numpy.nanmean(dam_results, axis=1)
                ax.plot(xs, mod_dam_ys, label=case.replace('_', ' '))
            ax.set_xlim(left=-.5, right=len(splits)-.5)
            ax.set_ylim(bottom=-0.02, top=0.1)
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
                      y=[-0.02, 0.02, 0.04, 0.06, 0.08, ],
                      xmin=-.2,
                      xmax=len(splits)-.8,
                      alpha=0.1,
                      linestyles='--',
                      )
            ax.legend()
            file_out = os.path.join(
                            plots_folder, 
                            'undamaged.jpeg',
                            )
            print(file_out)
            pyplot.savefig(file_out)
            pyplot.clf()
            pyplot.close()
            counter.update(1)
