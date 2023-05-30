import matplotlib.pyplot as plt
import prettyplot
import numpy as np
import os
import sklearn.metrics
import scipy.stats as stats

paths = [
    '../SimulationData/Ca_Integrals_for_ROC_plots/6Hz_1s.npz',
    '../SimulationData/Ca_Integrals_for_ROC_plots/6Hz_1s_1ms_jitter.npz',
    '../SimulationData/Ca_Integrals_for_ROC_plots/6Hz_1s_2ms_jitter.npz',
]

def kernel_density_estimation(x, dataset):
    kernel = stats.gaussian_kde(dataset)
    return kernel(x)

q = np.load('../SimulationData/thresholds.npz')
thresholds = q['thresholds']

def get_data_from_file(path):
    a = np.load(path)
    special_spine = a['spines']
    gs = a['gs']
    gs_non_matched_spine = []
    gs_matched_spine = []

    for i in range(gs.shape[0]):
        spine_index = (special_spine[i] // 10) * 10
        gs_non_matched_spine.append(np.concatenate([gs[i, :spine_index:10], gs[i, spine_index + 10::10]], axis=0))
        gs_matched_spine.append(gs[i, spine_index])

    return np.array(gs_non_matched_spine), np.array(gs_matched_spine)

data_random_spines, data_matched_spines = get_data_from_file(paths[1])

def kde(i):

    data_random_spines, data_matched_spines = get_data_from_file(paths[i])
    w = [1, 1, 1][i]


    a = np.reshape(data_random_spines, [-1])/thresholds[w]
    b = np.reshape(data_matched_spines, [-1])/thresholds[w]
    max_g = np.maximum(np.max(a), np.max(b))
    x = np.linspace(0, 4, 1000)
    y_a = kernel_density_estimation(x, a)
    y_b = kernel_density_estimation(x, b)

    return x, y_a, y_b

prettyplot.figure_with_specified_size((4, 4), (1, 1), (1.5, 1.5))

y_as = []
y_bs = []
for i in range(3):
    x, y_a, y_b = kde(i)
    y_as.append(y_a)
    y_bs.append(y_b)

plt.plot(x, y_a, color=prettyplot.colors['red'])
plt.fill_between(x, y_a, 0, color=prettyplot.colors['red'], alpha=0.333)

for i, y_b in enumerate(y_bs):
    plt.plot(x, y_b, color=prettyplot.color_list[i], label=str(i))
    #plt.fill_between(x, y_b, 0, color=prettyplot.color_list[i], alpha=0.333)

prettyplot.x_axis_only()
plt.xlim([-.1, 4])
plt.ylim([0, 1.5])
plt.legend()
plt.axvline(1, color='k', linewidth=2)
prettyplot.xlabel('g/g_threshold')
jitter = [0, 1, 2]
plt.savefig('../Figures/ROCPlots/gHistograms_1s_various_jitters.pdf', transparent=True)
plt.show()



def make_roc(negatives, positives, color='k', label=None, include_45_line=True):
    y_score = np.concatenate([negatives, positives])
    y_true = np.concatenate([np.zeros(shape=negatives.shape), np.ones(shape=positives.shape)], axis=0)
    sample_weight = np.concatenate([np.ones(shape=negatives.shape)/len(negatives), np.ones(shape=positives.shape)/len(positives)], axis=0)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score, sample_weight=sample_weight, pos_label=1)

    i = np.argmin(np.square(fpr - 0.10))

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
    data_random_spines_flattened = np.reshape(data_random_spines, [-1])
    data_random_spines_max = np.max(data_random_spines, axis=1)
    make_roc(data_random_spines_max, np.reshape(data_matched_spines, [-1]), color=prettyplot.color_list[i], label=[0, 1, 2][i], include_45_line=(i==0))

plt.legend(frameon=False)
plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.5, 1])
plt.ylim([0, 1.1])
plt.xlim([0, 1.1])
plt.savefig('../Figures/ROCPlots/ROCPlot_jitter.pdf', transparent=True)
plt.show()

