"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

This script collects the data for Figure 4D.
"""

import numpy as np
import pickle
import HocPythonTools
from Models.completeneuron import Neuron
from helper_functions import generate_spike_train_array, generate_spike_train
import datetime


optimum_offset = 7

matching_time = 1
control_time = 1
gap_time = 0.5

number_of_postsynaptic_spikes = int(np.round(matching_time * 8))


thresholds = {
    0.5:    0.05297869411632677,
    1:      0.07983869416080833,
    2:      0.12265248791940773
}

threshold = thresholds[1]

x = np.load('../SimulationData/time_delays.npz')
time_delays = x['time_delays']
seed = 2
print('numpy seed =', seed)
np.random.seed(seed)

# Set up the hoc interpreter and load the mechanisms
h = HocPythonTools.setup_neuron('../ChannelModFiles/x86_64/.libs/libnrnmech.so')
model = Neuron(h, include_ca_dependent_potentiation=True)

for branch in model.spiny_branches:
    for ampar in branch.ampa_list:
        ampar.max_potentiation = 10
        ampar.ca_potentiation_threshold = threshold
        ampar.sigmoid_sharpness = 100
        ampar.tau_g_dynamic_delayed = 500

# Parameters of simulation:
number_of_spines = model.number_of_spines
synapses_per_axon = 10
special_spine_1 = (np.random.randint(0, number_of_spines) // 10) * 10

special_spine_2_found = False
while not special_spine_2_found:
    special_spine_2 = (np.random.randint(0, number_of_spines) // 10) * 10
    if special_spine_2 != special_spine_1:
        special_spine_2_found = True



simulation_time_ms = 3000
spike_rate_kHz = 0.008
interspike_refactory_period_ms = 50


print('number_of_spines = ', number_of_spines)
print('Special spine =', special_spine_1)


presynaptic_stimulation_times = generate_spike_train_array(
    number_of_spines,
    [250, 250 + 1000*(matching_time + gap_time + control_time)],
    spike_rate_kHz,
    recovery_time=interspike_refactory_period_ms,
    synapses_per_axon=synapses_per_axon,
    multi_synapse_arangement='neighbors'
)

for i in range(len(presynaptic_stimulation_times)):
    presynaptic_stimulation_times[i] = presynaptic_stimulation_times[i][np.logical_or(presynaptic_stimulation_times[i] <= 250 + matching_time*1000, presynaptic_stimulation_times[i] >= 250 + 1000*(matching_time + gap_time))]

found_post_synaptic_times = False
while not found_post_synaptic_times:
    postsynaptic_stimulation_times = generate_spike_train(
        [250, 250 + matching_time*1000],
        spike_rate=spike_rate_kHz,
        recovery_time=interspike_refactory_period_ms,
    )
    if len(postsynaptic_stimulation_times) == number_of_postsynaptic_spikes:
        found_post_synaptic_times = True

special_spine_1_stimulation_times = generate_spike_train(
    [250 + 1000*(matching_time + gap_time), 250 + 1000*(matching_time + gap_time + control_time)],
    spike_rate=spike_rate_kHz,
    recovery_time=interspike_refactory_period_ms,
)

special_spine_2_stimulation_times = generate_spike_train(
    [250 + 1000*(matching_time + gap_time), 250 + 1000*(matching_time + gap_time + control_time)],
    spike_rate=spike_rate_kHz,
    recovery_time=interspike_refactory_period_ms,
)

special_spine_1_stimulation_times = np.concatenate([postsynaptic_stimulation_times - optimum_offset + time_delays[special_spine_1], special_spine_1_stimulation_times], 0)
special_spine_2_stimulation_times = np.concatenate([postsynaptic_stimulation_times - optimum_offset + time_delays[special_spine_2], special_spine_2_stimulation_times], 0)


for i in range(special_spine_1, special_spine_1 + 10):
    presynaptic_stimulation_times[i] = special_spine_1_stimulation_times.copy()

for i in range(special_spine_2, special_spine_2 + 10):
    presynaptic_stimulation_times[i] = special_spine_2_stimulation_times.copy()

iclamps = model.axosomatic_compartments.add_iclamps(postsynaptic_stimulation_times, 1, 4)

model.add_presynaptic_stimulation(presynaptic_stimulation_times)

scale = 0.08
model.set_AMPA_weight('all', 0.1 * scale)  # 0.1 for 50 synapses
model.set_NMDA_weight('all', 0.00005 * scale)  # 0.002 for 50 synapses

# Run the simulation

t = h.Vector().record(h._ref_t)

h.finitialize(-75)

print('Running Simulation')
h.continuerun(500 + 1000*(matching_time + gap_time + control_time))
print('Simulation Complete')

t = np.array(t)
v_spine = np.array(model.v_spine)
ca_spine = np.array(model.ca_spine)
g_spine = np.array(model.g_spine)
v_apical = np.array(model.v_apical_dendrite_shaft)
v_soma = np.array(model.v_soma)


save_dict = {
    'Description': 'Run of full model with two matched spines',
    'date': datetime.datetime.now(),
    't': t,
    'v_spine': v_spine,
    'ca_spine': ca_spine,
    'g_spine': g_spine,
    'v_apical': v_apical,
    'v_soma': v_soma,
    'postsynaptic_stimulation_times': postsynaptic_stimulation_times,
    'presynaptic_stimulation_times': presynaptic_stimulation_times,
    'special_spine_1': special_spine_1,
    'special_spine_2': special_spine_2,

    'number_of_spines': number_of_spines,
    'synapses_per_axon': synapses_per_axon,
    'threshold': threshold,
    'number_of_postsynaptic_spikes': number_of_postsynaptic_spikes,
    'seed': seed,
}


file_save_name = '../SimulationData/TwoSpinePotentiationExample.obj'

with open(file_save_name, 'wb') as handle:
    pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
