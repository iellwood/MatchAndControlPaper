"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

This script collects the data used for Figure 4, panel E, which show basic examples
of the control phase of the match and control principle.

Note that you must run this script with a specified seed (0) and then run
"FigureScripts/Figure_4E_PlotFailedPotentiationExample.py" to produce the figure in
"Figures/Potentiation_Examples"
"""

import numpy as np
import pickle
import HocPythonTools
from Models.completeneuron import Neuron
from helper_functions import generate_spike_train_array, generate_spike_train
import datetime

seed = 0

optimum_offset = 7
match_window_s = 1
match_window_size = match_window_s * 1000
number_of_postsynaptic_spikes = 4
control_window_size = 1000
gap_window_size = 500

q = np.load('../SimulationData/thresholds.npz')
thresholds = q['thresholds']
mw_dict = {0.5: 0, 1:1, 2:2}
threshold = thresholds[mw_dict[match_window_s]]

x = np.load('../SimulationData/time_delays.npz')
time_delays = x['time_delays']
print('numpy seed =', seed)
np.random.seed(seed)

# Set up the hoc interpreter and load the mechanisms
h = HocPythonTools.setup_neuron('../ChannelModFiles/x86_64/.libs/libnrnmech.so')
model = Neuron(h, include_ca_dependent_potentiation=True)

for branch in model.spiny_branches:
    for ampar in branch.ampa_list:
        ampar.ca_potentiation_threshold = threshold

# Parameters of simulation:
number_of_spines = model.number_of_spines
synapses_per_axon = 10
special_spine = (np.random.randint(0, number_of_spines) // 10) * 10

alternate_spine = (special_spine + (number_of_spines//2)) % number_of_spines
simulation_time_ms = 2750
spike_rate_kHz = 0.006
interspike_refactory_period_ms = 50

print('number_of_spines = ', number_of_spines)
print('Special spine =', special_spine)

presynaptic_stimulation_times = generate_spike_train_array(
    number_of_spines,
    [250, 250 + match_window_size + 500 + control_window_size],
    spike_rate_kHz,
    recovery_time=interspike_refactory_period_ms,
    synapses_per_axon=synapses_per_axon,
    multi_synapse_arangement='neighbors'
)

for i in range(len(presynaptic_stimulation_times)):
    presynaptic_stimulation_times[i] = presynaptic_stimulation_times[i][np.logical_or(presynaptic_stimulation_times[i] <= 250 + match_window_size, presynaptic_stimulation_times[i] >= 250 + match_window_size + gap_window_size)]


found_post_synaptic_times = False
while not found_post_synaptic_times:
    postsynaptic_stimulation_times = generate_spike_train(
        [250, 250 + match_window_size],
        spike_rate=spike_rate_kHz,
        recovery_time=interspike_refactory_period_ms,
    )
    if len(postsynaptic_stimulation_times) == number_of_postsynaptic_spikes:
        found_post_synaptic_times = True


special_spine_stimulation_times = generate_spike_train(
    [250 + match_window_size + gap_window_size , 250 + match_window_size + gap_window_size + control_window_size],
    spike_rate=spike_rate_kHz,
    recovery_time=interspike_refactory_period_ms,
)

special_spine_stimulation_times = np.concatenate([postsynaptic_stimulation_times - optimum_offset + time_delays[special_spine], special_spine_stimulation_times], 0)


for i in range(special_spine, special_spine + 10):
    presynaptic_stimulation_times[i] = special_spine_stimulation_times.copy()

iclamps = model.axosomatic_compartments.add_iclamps(postsynaptic_stimulation_times, 1, 4)

model.add_presynaptic_stimulation(presynaptic_stimulation_times)

# Run the simulation

t = h.Vector().record(h._ref_t)

h.finitialize(-75)
print('Running Simulation')
h.continuerun(250 + match_window_size + gap_window_size + control_window_size + 250)
print('Simulation Complete')

t = np.array(t)
v_spine = np.array(model.v_spine)
ca_spine = np.array(model.ca_spine)
g_spine = np.array(model.g_spine)
v_apical = np.array(model.v_apical_dendrite_shaft)
v_soma = np.array(model.v_soma)

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


save_dict = {
    'Description': 'Run of full model with potentiating spines',
    'date': datetime.datetime.now(),
    't': t,
    'v_special_spine': v_spine[special_spine, :],
    'ca_special_spine': ca_spine[special_spine, :],
    'g_special_spine': g_spine[special_spine, :],

    'v_second_best_spine': v_spine[second_best_spine, :],
    'ca_second_best_spine': ca_spine[special_spine, :],
    'g_second_best_spine': g_spine[second_best_spine, :],

    'v_apical': v_apical,
    'v_soma': v_soma,
    'postsynaptic_stimulation_times': postsynaptic_stimulation_times,
    'presynaptic_stimulation_times': presynaptic_stimulation_times,

    'special_spine': special_spine,
    'second_best_spine': second_best_spine,
    'number_of_spines': number_of_spines,
    'synapses_per_axon': synapses_per_axon,
    'threshold': threshold,
    'number_of_postsynaptic_spikes': number_of_postsynaptic_spikes,
    'seed': seed,
    'match_window_size': match_window_size,
    'control_window_size': control_window_size,
    'gap_window_size': gap_window_size,
}


file_save_name = '../SimulationData/FailedPotentiationExample_seed_' + str(seed) + '.obj'

with open(file_save_name, 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
