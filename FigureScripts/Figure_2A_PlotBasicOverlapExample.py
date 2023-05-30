"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

Produces Figure 2A
"""

import numpy as np
import matplotlib.pyplot as plt
import prettyplot
from helper_functions import decimate_trace
path = '../SimulationData/ExampleMatchingRun.npz'

data = np.load(path, allow_pickle=True)

t = data['t']
v_soma = data['v_soma']
v_spine = data['v_spine']
ca_spine = data['ca_spine']
g_spine = data['g_spine']
special_spine = data['special_spine']
postsynaptic_stimulation_times = data['postsynaptic_stimulation_times']
presynaptic_stimulation_times = data['presynaptic_stimulation_times']

# decimate the time series so that the plots don't have an excess number of points
skip = 10
v_spine = decimate_trace(t, v_spine, skip, axis=1)
v_soma = decimate_trace(t, v_soma, skip)
ca_spine = decimate_trace(t, ca_spine, skip, axis=1)
g_spine = decimate_trace(t, g_spine, skip, axis=1)
t = t[::skip]
t = t - 50 # make t = 0 the moment when the stimulation starts

########################################################################################################################
# Make the Plots
########################################################################################################################

t = np.array(t)
fig, axes = plt.subplots(7, 2, figsize=(5.2, 4))

# Matched Spine:

for i in range(postsynaptic_stimulation_times.shape[0]):
    axes[0, 0].axvline(postsynaptic_stimulation_times[i] - 50, color=prettyplot.colors['red'])
for i in range(presynaptic_stimulation_times[special_spine].shape[0]):
    axes[1, 0].axvline(presynaptic_stimulation_times[special_spine][i] - 50, color=prettyplot.colors['blue'])
axes[2, 0].plot(t, v_soma)
axes[2, 0].set_ylim([-90, 50])
axes[3, 0].plot(t, v_spine[special_spine])
axes[3, 0].set_ylim([-90, 50])
axes[4, 0].plot(t, ca_spine[special_spine])
axes[5, 0].plot(t, np.power(ca_spine[special_spine], 4))
axes[6, 0].plot(t, g_spine[special_spine])

# Unmatched Spine:

alternate_spine = (750 + special_spine) % 1500

for i in range(postsynaptic_stimulation_times.shape[0]):
    axes[0, 1].axvline(postsynaptic_stimulation_times[i] - 50, color=prettyplot.colors['red'])
for i in range(presynaptic_stimulation_times[alternate_spine].shape[0]):
    axes[1, 1].axvline(presynaptic_stimulation_times[alternate_spine][i] - 50, color=prettyplot.colors['blue'])

axes[2, 1].plot(t, v_soma)
axes[2, 1].set_ylim([-85, 50])

axes[3, 1].plot(t, v_spine[alternate_spine])
axes[3, 1].set_ylim([-85, 50])
axes[4, 1].plot(t, ca_spine[alternate_spine])
axes[5, 1].plot(t, np.power(ca_spine[alternate_spine], 4))
axes[6, 1].plot(t, g_spine[alternate_spine])



max_ca = np.maximum(np.max(ca_spine[special_spine]), np.max(ca_spine[alternate_spine]))
axes[4, 0].set_ylim([0, max_ca*1.1])
axes[4, 1].set_ylim([0, max_ca*1.1])

max_ca_4 = np.maximum(np.max(np.power(ca_spine[special_spine], 4)), np.max(np.power(ca_spine[alternate_spine], 4)))
axes[5, 0].set_ylim([0, max_ca_4*1.1])
axes[5, 1].set_ylim([0, max_ca_4*1.1])

max_g = np.maximum(np.max(g_spine[special_spine]), np.max(g_spine[alternate_spine]))
axes[6, 0].set_ylim([0, max_g*1.1])
axes[6, 1].set_ylim([0, max_g*1.1])

# Cleanup:

for i in range(len(axes) - 1):
    for j in range(len(axes[i])):
        axes[i, j].axis('off')


axes[3, 0].plot([10, 10], [0, 50], color='k', linewidth=2)

prettyplot.x_axis_only(axes[-1, 0])
prettyplot.x_axis_only(axes[-1, 1])

for i in range(len(axes)):
    for j in range(len(axes[i])):
        axes[i, j].set_xlim([0, 1100])

plt.savefig('../Figures/Figure_2A_BasicSpikeTrainMatchingExample.pdf')
plt.show()

