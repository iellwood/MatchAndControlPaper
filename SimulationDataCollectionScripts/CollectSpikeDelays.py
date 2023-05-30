"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

This script simulates a single spike propagating through the dendritic tree. The data is used
to compute the transmission delays for each spine.
"""

import numpy as np
import pickle
import HocPythonTools
from Models.completeneuron import Neuron
from helper_functions import generate_spike_train_array, generate_spike_train
import datetime

h = HocPythonTools.setup_neuron('../ChannelModFiles/x86_64/.libs/libnrnmech.so')

np.random.seed(0)
def collect_data():
    # Set up the hoc interpreter and load the mechanisms
    model = Neuron(h, include_ca_dependent_potentiation=True)

    # Parameters of simulation:
    number_of_spines = model.number_of_spines
    synapses_per_axon = 10
    simulation_time_ms = 250 + 100
    spike_rate_kHz = 0.006
    interspike_refactory_period_ms = 50
    print('number_of_spines = ', number_of_spines)

    presynaptic_stimulation_times = generate_spike_train_array(
        number_of_spines,
        [1, 349],
        spike_rate_kHz,
        recovery_time=interspike_refactory_period_ms,
        synapses_per_axon=synapses_per_axon,
        multi_synapse_arangement='neighbors'
    )

    postsynaptic_stimulation_times = np.array([250.0])

    iclamps = model.axosomatic_compartments.add_iclamps(postsynaptic_stimulation_times, 1, 4)

    model.add_presynaptic_stimulation(presynaptic_stimulation_times)

    # Run the simulation

    t = h.Vector().record(h._ref_t)

    h.finitialize(-70)

    print('Running Simulation')
    h.continuerun(simulation_time_ms)
    print('Simulation Complete')

    t = np.array(t)
    v_spine = np.array(model.v_spine)
    ca_spine = np.array(model.ca_spine)
    g_spine = np.array(model.g_spine)
    v_apical = np.array(model.v_apical_dendrite_shaft)
    v_soma = np.array(model.v_soma)
    print('v_apical.shape =', v_apical.shape)
    print('v_spine.shape =', v_spine.shape)

    distances = []
    for i in range(number_of_spines):
        branch_number = model.branch_ID[i]
        branch_spine_number = model.branch_spine_number[i]
        distances.append(h.distance(model.spiny_branches[branch_number].spine_list[branch_spine_number].head(0.5)))


    save_dict = {
        'Description': 'Run of full model with potentiating spines',
        'date': datetime.datetime.now(),
        't': t,
        'v_spine': v_spine,
        'ca_spine': ca_spine,
        'g_spine': g_spine,
        'v_apical': v_apical,
        'v_soma': v_soma,
        'postsynaptic_stimulation_times': postsynaptic_stimulation_times,
        'presynaptic_stimulation_times': presynaptic_stimulation_times,
        'number_of_spines': number_of_spines,
        'synapses_per_axon': synapses_per_axon,
        'distances': np.array(distances),
    }

    return save_dict

data_list = []
for i in range(10):
    print('run #', i)
    data_list.append(collect_data())

file_save_name = '../SimulationData/spike_delay_data.obj'

with open(file_save_name, 'wb') as handle:
    pickle.dump(data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
