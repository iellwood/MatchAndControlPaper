"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

This script collects the data used for Figure 5.
"""

import numpy as np
import HocPythonTools
from Models.completeneuron import Neuron
from helper_functions import generate_spike_train_array, generate_spike_train
from scipy.signal import argrelextrema
import os
import time
from parallel_run import ParallelRun

max_number_of_processes = 12

optimum_offset = 7
match_window_s = 0.5
match_window = 1000*match_window_s # ms
match_window_size = match_window
gap_window_size = 500
control_window_size = 1000

path = '../SimulationData/ControlPhaseTests/6Hz_' + str(match_window_s) + 's'

q = np.load('../SimulationData/thresholds.npz')
thresholds = q['thresholds']
mw_dict = {0.5: 0, 1:1, 2:2}
threshold = thresholds[mw_dict[match_window_s]]

def argument_function(i):
    return ( )

def get_control_phase_spike_outputs():

    # This line is essential to get the processes to have different random seeds
    np.random.seed((os.getpid() * int(time.time()*10000)) % 123456789)

    x = np.load('../SimulationData/time_delays.npz')
    time_delays = x['time_delays']

    # Set up the hoc interpreter and load the mechanisms
    h = HocPythonTools.setup_neuron('../ChannelModFiles/x86_64/.libs/libnrnmech.so')
    model = Neuron(h, include_ca_dependent_potentiation=True)

    # Set the ca-integral potentiation threshold
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

    presynaptic_stimulation_times = generate_spike_train_array(
        number_of_spines,
        [250, 250 + match_window_size + gap_window_size + control_window_size],
        spike_rate_kHz,
        recovery_time=interspike_refactory_period_ms,
        synapses_per_axon=synapses_per_axon,
        multi_synapse_arangement='neighbors'
    )

    for i in range(len(presynaptic_stimulation_times)):
        presynaptic_stimulation_times[i] = presynaptic_stimulation_times[i][np.logical_or(presynaptic_stimulation_times[i] <= 250 + match_window_size, presynaptic_stimulation_times[i] >= 250 + match_window_size + gap_window_size)]

    postsynaptic_stimulation_times = generate_spike_train(
        [250, 250 + match_window_size],
        spike_rate=spike_rate_kHz,
        recovery_time=interspike_refactory_period_ms,
    )

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
    h.continuerun(250 + match_window_size + gap_window_size + control_window_size + 250)

    t = np.array(t)
    v_soma = np.array(model.v_soma)

    local_maxima = argrelextrema(v_soma, np.greater)[0]
    local_maxima = local_maxima[v_soma[local_maxima] > 0]
    somatic_spike_times = t[local_maxima]

    return {
            'presynaptic_stimulation_times': presynaptic_stimulation_times,
            'postsynaptic_stimulation_times': postsynaptic_stimulation_times,
            'somatic_spike_times': somatic_spike_times,
            'special_spine': special_spine,
            }


if __name__ == '__main__':

    parallel_run = ParallelRun(
        target_function=get_control_phase_spike_outputs,
        argument_function=argument_function,
        max_number_of_processes=max_number_of_processes,
        iterations=1000,
    )

    outputs = parallel_run.run()

    somatic_spike_times = []
    presynaptic_stimulation_times = []
    postsynaptic_stimulation_times = []
    special_spines = []

    for output in outputs:
        presynaptic_stimulation_times.append(output['presynaptic_stimulation_times'])
        postsynaptic_stimulation_times.append(output['postsynaptic_stimulation_times'])
        somatic_spike_times.append(output['somatic_spike_times'])
        special_spines.append(output['special_spine'])


    np.savez(
        path,
        match_window=match_window,
        special_spines=special_spines,
        somatic_spike_times=somatic_spike_times,
        presynaptic_stimulation_times=presynaptic_stimulation_times,
        postsynaptic_stimulation_times=postsynaptic_stimulation_times
    )