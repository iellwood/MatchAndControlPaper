"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

This script collects the data for Figure 2A
"""

import numpy as np
import HocPythonTools
import time
import os
import prettyplot
import matplotlib.pyplot as plt
from parallel_run import ParallelRun

optimum_offset = 6.749648449404958
match_window_s = 1
match_window = 1000*match_window_s # ms

path = '../SimulationData/MatchingExample'

np.random.seed(0)

special_spine= np.random.randint(0, 1500)

x = np.load('../SimulationData/time_delays.npz')
time_delays = x['time_delays']

from helper_functions import generate_spike_train_array, generate_spike_train
from Models.completeneuron import Neuron

# Initialize the hoc interpreter and load all mechanisms
# Set up the hoc interpreter and load the mechanisms
h = HocPythonTools.setup_neuron('../ChannelModFiles/x86_64/.libs/libnrnmech.so')
model = Neuron(h, include_ca_dependent_potentiation=False)

# Parameters of simulation:
number_of_spines = model.number_of_spines
synapses_per_axon = 10

simulation_time_ms = match_window
presynaptic_spike_rate_kHz = 0.008
interspike_refactory_period_ms = 50

presynaptic_stimulation_times = generate_spike_train_array(
    number_of_spines,
    [50, simulation_time_ms + 50],
    presynaptic_spike_rate_kHz,
    recovery_time=interspike_refactory_period_ms,
    synapses_per_axon=synapses_per_axon,
    multi_synapse_arangement='neighbors'
)

presynaptic_stimulation_times_special_spine = presynaptic_stimulation_times[special_spine]
post_synaptic_stimulation_times = presynaptic_stimulation_times_special_spine + optimum_offset - time_delays[special_spine]

model.axosomatic_compartments.add_iclamps(post_synaptic_stimulation_times, 1, 4)
model.add_presynaptic_stimulation(presynaptic_stimulation_times)

t = h.Vector().record(h._ref_t)
h.finitialize(-75)
h.continuerun(simulation_time_ms + 150)

np.savez('../SimulationData/ExampleMatchingRun',
            t=np.array(t),
            v_spine=np.array(model.v_spine),
            v_soma=np.array(model.v_soma),
            g_spine=np.array(model.g_spine),
            ca_spine=np.array(model.ca_spine),
            presynaptic_stimulation_times=presynaptic_stimulation_times,
            postsynaptic_stimulation_times=post_synaptic_stimulation_times,
            special_spine=special_spine,
         )

