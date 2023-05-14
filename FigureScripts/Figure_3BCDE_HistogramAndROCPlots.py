"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

Produces Figure 3, panels B, C, D & E
"""

import matplotlib.pyplot as plt
import prettyplot
import numpy as np
import os
import sklearn.metrics
import scipy.stats as stats

paths = [
    '../SimulationData/Ca_Integrals_for_ROC_plots/8Hz_0.5s.npz',
    '../SimulationData/Ca_Integrals_for_ROC_plots/8Hz_1s_0s_jitter.npz',
    '../SimulationData/Ca_Integrals_for_ROC_plots/8Hz_2s.npz',
]

thresholds = {
    0.5:    0.05297869411632677,
    1:      0.0733648827356375,
    2:      0.12265248791940773
}

def kernel_density_estimation(x, dataset):
    kernel = stats.gaussian_kde(dataset)
    return kernel(x)

def get_data_from_file(path):
    a = np.load(path)
    special_spine = a['spines']
    offset_time = a['optimum_offset']
    gs = a['gs']
    gs_non_matched_spine = []
    gs_matched_spine = []

    for i in range(gs.shape[0]):
        spine_index = (special_spine[i] // 10) * 10
        gs_non_matched_spine.append(np.concatenate([gs[i, :spine_index:10], gs[i, spine_index + 10::10]], axis=0))
        gs_matched_spine.append(gs[i, spine_index])

    return np.array(gs_non_matched_spine), np.array(gs_matched_spine)

data_random_spines, data_matched_spines = get_data_from_file(paths[1])

def plot_hist(i):

    data_random_spines, data_matched_spines = get_data_from_file(paths[i])
    w = [0.5, 1, 2][i]

    prettyplot.figure_with_specified_size((4, 4), (1, 1), (1.5, 1.5))

    a = np.reshape(data_random_spines, [-1])/thresholds[w]
    b = np.reshape(data_matched_spines, [-1])/thresholds[w]
    max_g = np.maximum(np.max(a), np.max(b))
    x = np.linspace(0, max_g*1.2, 1000)
    y_a = kernel_density_estimation(x, a)
    y_b = kernel_density_estimation(x, b)
    plt.plot(x, y_a, color=prettyplot.colors['red'])
    plt.plot(x, y_b, color=prettyplot.colors['blue'])
    plt.fill_between(x, y_a, 0, color=prettyplot.colors['red'], alpha=0.333)
    plt.fill_between(x, y_b, 0, color=prettyplot.colors['blue'], alpha=0.333)

    prettyplot.x_axis_only()
    plt.xlim([-.1, 3])
    plt.ylim([0, 1.5])
    plt.axvline(1, color='k', linewidth=2)
    prettyplot.xlabel('g/g_threshold')
    prettyplot.title('match window = ' + str(w))
    plt.savefig('../Figures/ROCPlots/gHistograms_' + str(w) + 's.pdf', transparent=True)
    plt.show()

plot_hist(0)
plot_hist(1)
plot_hist(2)


def make_roc(negatives, positives, color='k', label=None, include_45_line=True):
    y_score = np.concatenate([negatives, positives])
    y_true = np.concatenate([np.zeros(shape=negatives.shape), np.ones(shape=positives.shape)], axis=0)
    sample_weight = np.concatenate([np.ones(shape=negatives.shape)/len(negatives), np.ones(shape=positives.shape)/len(positives)], axis=0)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score, sample_weight=sample_weight, pos_label=1)

    i = np.argmin(np.square(fpr - 0.001486517711815405))

    print('for fpr =', fpr[i], 'true positive rate =', tpr[i], 'threshold =', thresholds[i])

    plt.plot(fpr, tpr, label=label, color=color)
    if include_45_line:
        plt.plot([0, 1], [0, 1], color='k')
    prettyplot.no_box()
    plt.gca().set_aspect(1)
    prettyplot.xlabel('false positive rate')
    prettyplot.ylabel('true positive rate')
    #plt.show()

prettyplot.figure_with_specified_size((4, 4), (1, 1), (1.5, 1.5))

for i in range(len(paths)):
    data_random_spines, data_matched_spines = get_data_from_file(paths[i])
    make_roc(np.reshape(data_random_spines, [-1]), np.reshape(data_matched_spines, [-1]), color=prettyplot.color_list[i], label=[0.5, 1, 2][i], include_45_line=(i==0))

plt.legend(frameon=False)
plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.5, 1])
plt.ylim([0, 1.1])
plt.xlim([0, 1.1])
plt.savefig('../Figures/ROCPlots/ROCPlot.pdf', transparent=True)
plt.show()

