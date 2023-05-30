"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

Produces The panels of Figure 5
"""

import numpy as np
import matplotlib.pyplot as plt
import prettyplot

data = np.load('../SimulationData/ControlPhaseTests/failed_and_spurious_spikes.npz', allow_pickle=True)
fraction_failed_spikes = data['fraction_failed_spikes']
fraction_spurious_spikes = data['fraction_spurious_spikes']
number_of_stims = data['number_of_stims']
match_windows = data['match_windows']

# Figure 5A
prettyplot.figure_with_specified_size((5, 4), (1, 0.5), (1, 2))
ffs = []
fss = []
ffs_sem = []
fss_sem = []
for i in range(3):
    ffs.append(np.mean(fraction_failed_spikes[i]))
    fss.append(np.mean(fraction_spurious_spikes[i]))
    ffs_sem.append(np.std(fraction_failed_spikes[i])/np.sqrt(len(fraction_failed_spikes[i])))
    fss_sem.append(np.std(fraction_spurious_spikes[i])/np.sqrt(len(fraction_spurious_spikes[i])))
plt.errorbar(np.arange(3), 1 - np.array(ffs), yerr=ffs_sem, marker='o', color=prettyplot.colors['blue'])
plt.errorbar(np.arange(3), fss, yerr=fss_sem,  marker='o', color=prettyplot.colors['red'])
plt.ylim([0, 1])
prettyplot.no_box()
plt.xticks([0, 1, 2], ['0.5', '1', '2'])
plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
prettyplot.xlabel('match window s')
plt.savefig('../Figures/Figure_5A_fraction_of_spurious_and_failed_spikes.pdf', transparent=True)
plt.show()

# Figure 5B
prettyplot.figure_with_specified_size((5, 4), (1, 0.5), (2, 2))
for i in range(3):
    counts = np.unique(number_of_stims[i])
    print(counts)
    if counts[0] == 0:
        counts = counts[1:]
    frac_fail = []
    frac_fail_sem = []
    count_list = []
    for j in range(len(counts)):
        q = number_of_stims[i] == int(counts[j])
        if np.sum(q) >= 5:
            frac_fail.append(np.mean(fraction_failed_spikes[i][q]))
            frac_fail_sem.append(np.std(fraction_failed_spikes[i][q]) / np.sqrt(len(fraction_failed_spikes[i][q])))


    counts = counts[:len(frac_fail)]
    plt.errorbar(counts/match_windows[i], 1 - np.array(frac_fail), yerr=frac_fail_sem, color=prettyplot.color_list[i], marker='o', capsize=3)
plt.ylim([0, 1])
plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
prettyplot.no_box()
plt.axvline(6, color='k')
prettyplot.xlabel('query spike rate Hz')
plt.savefig('../Figures/Figure_5B_fraction_of_successful_spikes_by_frequency.pdf', transparent=True)

plt.show()

# Figure 5C
prettyplot.figure_with_specified_size((5, 4), (1, 0.5), (2, 2))
for i in range(3):
    counts = np.unique(number_of_stims[i])
    if counts[0] == 0:
        counts = counts[1:]
    frac_spur = []
    frac_spur_sem = []
    for j in range(len(counts)):
        q = number_of_stims[i] == int(counts[j])
        if np.sum(q) >= 10:
            frac_spur.append(np.mean(fraction_spurious_spikes[i][q]))
            frac_spur_sem.append(np.std(fraction_spurious_spikes[i][q]) / np.sqrt(len(fraction_spurious_spikes[i][q])))
    counts = counts[:len(frac_spur)]
    plt.errorbar(counts/match_windows[i], frac_spur, yerr=frac_spur_sem, color=prettyplot.color_list[i], marker='o', capsize=3)
plt.ylim([0, 1])
plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
plt.axvline(6, color='k')
prettyplot.no_box()
prettyplot.xlabel('query spike rate Hz')
plt.savefig('../Figures/Figure_5C_fraction_of_spurious_spikes_by_frequency.pdf', transparent=True)
plt.show()


