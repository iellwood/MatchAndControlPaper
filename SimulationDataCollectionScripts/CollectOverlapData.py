"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

This script collects the data used for figure 2, panels C & D

Note that the script must be run twice to produce the data from the paper, once with the
test_data flag set to true.

Also note that this file uses random number generation, so the exact findings from the paper
may be slightly altered.
"""


import numpy as np
import HocPythonTools
import time
import os
import prettyplot
import matplotlib.pyplot as plt
from parallel_run import ParallelRun

test_data = False
match_window_s = 1
match_window = 1000*match_window_s # ms
max_number_of_processes = 12

if test_data:
    path = '../SimulationData/OverlapData/TEST_DATA_6Hz_' + str(match_window_s) + 's'
else:
    path = '../SimulationData/OverlapData/6Hz_' + str(match_window_s) + 's'

def argument_function(i):
    return ( )

def get_g_dynamics():

    # This line is essential to get the processes to have different random seeds
    np.random.seed((os.getpid() * int(time.time()*10000)) % 123456789)

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

    presynaptic_spike_rate_kHz = 0.006
    interspike_refactory_period_ms = 50

    presynaptic_stimulation_times = generate_spike_train_array(
        number_of_spines,
        [50, match_window + 50],
        presynaptic_spike_rate_kHz,
        recovery_time=interspike_refactory_period_ms,
        synapses_per_axon=synapses_per_axon,
        multi_synapse_arangement='neighbors'
    )

    post_synaptic_stimulation_times = generate_spike_train(
        time_range=[50, match_window + 50],
        spike_rate=presynaptic_spike_rate_kHz,
        recovery_time=interspike_refactory_period_ms
    )


    model.axosomatic_compartments.add_iclamps(post_synaptic_stimulation_times, 1, 4)
    model.add_presynaptic_stimulation(presynaptic_stimulation_times)

    t = h.Vector().record(h._ref_t)
    h.finitialize(-75)
    h.continuerun(50 + match_window + 100)


    gs = np.array([g[-1] for g in model.g_spine])

    return {'gs': gs, 'presynaptic_stimulation_times': presynaptic_stimulation_times, 'postsynaptic_stimulation_times': post_synaptic_stimulation_times}



if __name__ == '__main__':

    parallel_run = ParallelRun(
        target_function=get_g_dynamics,
        argument_function=argument_function,
        max_number_of_processes=max_number_of_processes,
        iterations=100,
    )

    outputs = parallel_run.run()

    gs = []
    presynaptic_stimulation_times = []
    postsynaptic_stimulation_times = []

    for output in outputs:
        gs.append(output['gs'])
        presynaptic_stimulation_times.append(output['presynaptic_stimulation_times'])
        postsynaptic_stimulation_times.append(output['postsynaptic_stimulation_times'])

    gs = np.array(gs)

    np.savez(path, gs=gs, match_window=match_window, presynaptic_stimulation_times=presynaptic_stimulation_times, postsynaptic_stimulation_times=postsynaptic_stimulation_times)