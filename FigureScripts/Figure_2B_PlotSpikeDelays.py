"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

Produces Figure 2B
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import prettyplot
from helper_functions import find_maximum_with_second_order_approximation

file_save_name = '../SimulationData/spike_delay_data.obj'

with open(file_save_name, 'rb') as handle:
    data = pickle.load(handle)

def get_spike_delays_from_dataset(data):
    t = data['t']
    v_spine = data['v_spine']
    ca_spine = data['ca_spine']
    g_spine = data['g_spine']
    v_apical = data['v_apical']
    v_soma = data['v_soma']
    postsynaptic_stimulation_times = data['postsynaptic_stimulation_times']
    presynaptic_stimulation_times = data['presynaptic_stimulation_times']
    distances = data['distances']
    print('v_apical.shape =', v_apical.shape)
    print('v_spine.shape =', v_spine.shape)
    print('distances.shape =', distances.shape)

    ########################################################################################################################
    # Make the Plots
    ########################################################################################################################

    t = np.array(t)

    times = []
    for i in range(v_spine.shape[0]):

        v = v_spine[i, :]

        times.append(find_maximum_with_second_order_approximation(t, v))
    time_delays = np.array(times) - 250
    return distances, time_delays

time_delays = []
for d in data:
    distances, time_delays_one_run = get_spike_delays_from_dataset(d)
    time_delays.append(time_delays_one_run)

time_delays = np.mean(time_delays, axis=0)

np.savez('../SimulationData/time_delays', distances=distances, time_delays=time_delays)


prettyplot.figure_with_specified_size((5, 4), (1, 0.5), (1, 2))
plt.scatter(distances, time_delays, marker='.', s=1, color='k')
prettyplot.no_box()
prettyplot.xlabel('distance to soma um')
prettyplot.ylabel('time lag (ms)')
plt.xlim([500, 1000])
plt.ylim([0, 5])
plt.savefig('../Figures/Figure_2B_SpikeDelays.pdf')
plt.show()
