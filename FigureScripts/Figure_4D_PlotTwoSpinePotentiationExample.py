"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

Produces Figure 4D
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import prettyplot

file_save_name = '../SimulationData/TwoSpinePotentiationExample.obj'

with open(file_save_name, 'rb') as handle:
    data = pickle.load(handle)

t = data['t']
v_spine = data['v_spine']
ca_spine = data['ca_spine']
g_spine = data['g_spine']
v_apical = data['v_apical']
v_soma = data['v_soma']
special_spine_1 = data['special_spine_1']
special_spine_2 = data['special_spine_2']

threshold = data['threshold']
print('threshold =', threshold)
postsynaptic_stimulation_times = data['postsynaptic_stimulation_times']
presynaptic_stimulation_times = data['presynaptic_stimulation_times']
seed = data['seed']
print('v_apical.shape =', v_apical.shape)
print('v_spine.shape =', v_spine.shape)
print('special_spine_1 =', special_spine_1)
print('special_spine_2 =', special_spine_2)

########################################################################################################################
# Make the Plots
########################################################################################################################

t = np.array(t) - 250
fig, axes = plt.subplots(4, 3, figsize=(12, 2))

# Somatic plots:

for i in range(postsynaptic_stimulation_times.shape[0]):
    axes[0, 0].axvline(postsynaptic_stimulation_times[i] - 250, color=[1.0, 0.5, 0.5])
for i in range(presynaptic_stimulation_times[special_spine_1].shape[0]):
    axes[1, 0].axvline(presynaptic_stimulation_times[special_spine_1][i] - 250, color=prettyplot.colors['red'])
for i in range(presynaptic_stimulation_times[special_spine_2].shape[0]):
    axes[1, 0].axvline(presynaptic_stimulation_times[special_spine_2][i] - 250, color=prettyplot.colors['blue'])



axes[2, 0].plot(t, v_soma)

# Special spine plots

for i in range(postsynaptic_stimulation_times.shape[0]):
    axes[0, 1].axvline(postsynaptic_stimulation_times[i] - 250, color=[1.0, 0.5, 0.5])

for i in range(presynaptic_stimulation_times[special_spine_1].shape[0]):
    axes[1, 1].axvline(presynaptic_stimulation_times[special_spine_1][i] - 250, color=[0.5, 0.5, 1.0])

axes[2, 1].plot(t, v_spine[special_spine_1, :])
axes[3, 1].plot(t, g_spine[special_spine_1, :])
axes[3, 1].axhline(threshold, color=[0.8, 0.8, 0.8])
axes[3, 2].axhline(threshold, color=[0.8, 0.8, 0.8])

# alternate spine plots

for i in range(postsynaptic_stimulation_times.shape[0]):
    axes[0, 2].axvline(postsynaptic_stimulation_times[i] - 250, color=[1.0, 0.5, 0.5])

for i in range(presynaptic_stimulation_times[special_spine_2].shape[0]):
    axes[1, 2].axvline(presynaptic_stimulation_times[special_spine_2][i] - 250, color=[0.5, 0.5, 1.0])
axes[2, 2].plot(t, v_spine[special_spine_2, :])

axes[3, 2].plot(t, g_spine[special_spine_2, :], color='k')

g_max = np.maximum(np.max(g_spine[special_spine_1, :]), np.max(g_spine[special_spine_2, :]))

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

plt.savefig('../Figures/TwoSpineExample_seed_' + str(seed) + '.pdf', transparent=True)
plt.show()

