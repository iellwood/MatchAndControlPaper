"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

Produces Figure 4, panels B, E & F.

NOTE: Because the files produced by "SimulationDataCollectionScripts/PotentiationExample.py"
are quite large, this file only plots whatever is in the saved data file
"SimulationData/BasicPotentiationExample.obj"
This file uses the saved seed value in that file to save a figure with the approriate
name, such as "Figures/Potentiation_Examples/Potentiation_Example_seed_3.pdf"

Hence, you will have to run PotentiationExample.py five times alternating with running this
file in order to get all of the figures saved in "Figures/Potentiation_Examples/"
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import prettyplot

file_save_name = '../SimulationData/BasicPotentiationExample.obj'

with open(file_save_name, 'rb') as handle:
    data = pickle.load(handle)

t = data['t']
v_spine = data['v_spine']
ca_spine = data['ca_spine']
g_spine = data['g_spine']
v_apical = data['v_apical']
v_soma = data['v_soma']
special_spine = data['special_spine']
alternate_spine = data['alternate_spine']
threshold = data['threshold']
print('threshold =', threshold)
postsynaptic_stimulation_times = data['postsynaptic_stimulation_times']
presynaptic_stimulation_times = data['presynaptic_stimulation_times']
seed = data['seed']


def get_second_best_spine(time_for_comparison, special_spine, t, g_spine):
    i = np.argmin(np.abs(t - time_for_comparison))
    g = g_spine[:, i]
    g = g[::10]
    I = np.argsort(g)
    I = np.flip(I)
    if I[0] * 10 != special_spine:
        return I[0] * 10
    else:
        return I[1] * 10

second_best_spine = get_second_best_spine(1250, special_spine, t, g_spine)

alternate_spine = second_best_spine

########################################################################################################################
# Make the Plots
########################################################################################################################

t = np.array(t) - 250
fig, axes = plt.subplots(4, 3, figsize=(12, 2))

# Somatic plots:

for i in range(postsynaptic_stimulation_times.shape[0]):
    axes[0, 0].axvline(postsynaptic_stimulation_times[i] - 250, color=[1.0, 0.5, 0.5])
for i in range(presynaptic_stimulation_times[special_spine].shape[0]):
    axes[1, 0].axvline(presynaptic_stimulation_times[special_spine][i] - 250, color=[1.0, 0.5, 0.5])

axes[2, 0].plot(t, v_soma)

# Special spine plots

for i in range(postsynaptic_stimulation_times.shape[0]):
    axes[0, 1].axvline(postsynaptic_stimulation_times[i] - 250, color=[1.0, 0.5, 0.5])
for i in range(presynaptic_stimulation_times[special_spine].shape[0]):
    axes[1, 1].axvline(presynaptic_stimulation_times[special_spine][i] - 250, color=[0.5, 0.5, 1.0])

axes[2, 1].plot(t, v_spine[special_spine, :])
axes[3, 1].plot(t, g_spine[special_spine, :])
axes[3, 1].axhline(threshold, color=[0.8, 0.8, 0.8])
axes[3, 2].axhline(threshold, color=[0.8, 0.8, 0.8])

# alternate spine plots

for i in range(postsynaptic_stimulation_times.shape[0]):
    axes[0, 2].axvline(postsynaptic_stimulation_times[i] - 250, color=[1.0, 0.5, 0.5])
for i in range(presynaptic_stimulation_times[alternate_spine].shape[0]):
    axes[1, 2].axvline(presynaptic_stimulation_times[alternate_spine][i] - 250, color=[0.5, 0.5, 1.0])
axes[2, 2].plot(t, v_spine[alternate_spine, :])

axes[3, 2].plot(t, g_spine[alternate_spine, :],color='k')

g_max = np.max(g_spine[special_spine, :])

axes[3, 1].set_ylim([0, g_max*1.1])
axes[3, 2].set_ylim([0, g_max*1.1])




for i in range(len(axes)):
    for j in range(len(axes[i])):
        axes[i, j].set_xlim([t[0] + 200, t[-1] - 200])
        prettyplot.x_axis_only(axes[i, j])

for i in range(3):
    for j in range(3):
        axes[i, j].axis('off')


axes[2, 0].set_ylim([-100, 50])
axes[2, 1].set_ylim([-100, 50])
axes[2, 2].set_ylim([-100, 50])

plt.savefig('../Figures/Potentiation_Examples/Potentiation_Example_seed_' + str(seed) + '.pdf', transparent=True)
plt.show()

