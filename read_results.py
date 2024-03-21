import matplotlib
import numpy
import os
import scipy

from matplotlib import pyplot

results = dict()

with open(os.path.join('results/fernandino/all_results_opensubs_min10_1.txt')) as i:
    for l in i:
        line = l.strip().split('\t')
        dataset = int(line[0])
        if dataset not in results.keys():
            results[dataset] = dict()
        area = line[1]
        if area not in results[dataset].keys():
            results[dataset][area] = dict()
        model = line[2].replace('word2vec', 'gensim_w2v')
        scores = numpy.array(line[3:], dtype=numpy.float64)
        #results[dataset].append([(area, model), scores])
        results[dataset][area][model] = scores
colors = ['teal', 'orange', 'orchid', 'grey', 'dodgerblue']
### plotting
for dataset, dataset_data in results.items():
    plots = os.path.join('plots', 'fernandino', str(dataset))
    os.makedirs(plots, exist_ok=True)
    for area, area_data in dataset_data.items():
        fig, ax = pyplot.subplots(constrained_layout=True)
        ax.set_ylim(bottom=-.04, top=0.1)
        models = sorted(area_data.keys())
        #print(models)
        ax.hlines(
                  y=[0.0],
                  xmin=-.2,
                  xmax=len(models)-.8,
                  alpha=0.3,
                  linestyles='-',
                  color='grey',
                  )
        ax.hlines(
                  y=[-0.02, 0.02, 0.04, 0.06, 0.08, ],
                  xmin=-.2,
                  xmax=len(models)-.8,
                  alpha=0.1,
                  linestyles='--',
                  )
        plot_colors = {k : colors[k_i] for k_i, k in enumerate(models)}
        for m_i, m in enumerate(models):
            vi = ax.violinplot(
                          area_data[m],
                          [m_i],
                          showextrema=False,
                          )
            for b in vi['bodies']:
                vi_m = numpy.mean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], -numpy.inf, vi_m)
                b.set_color(plot_colors[m])
            ax.scatter(
                                   m_i,
                                   numpy.average(area_data[m]),
                                   color=plot_colors[m],
                                   edgecolors='white',
                                   marker='D',
                                   )
        pyplot.xticks(
                  ticks=range(len(models)), 
                  labels=[m[:10] for m in models]
                  )
        pyplot.title(area)
        pyplot.savefig(os.path.join(plots, '{}.jpg'.format(area)))
        pyplot.clf()
        pyplot.close()



