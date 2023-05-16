"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

This script collects the data used for the histogram and ROC plots in Figure 3

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

optimum_offset = 7
match_window_s = 1
match_window = 1000*match_window_s # ms
spike_position_jitter_std = 2
special_spines = np.arange(1500) # spines to run
max_number_of_processes = 12

path = '../SimulationData/Ca_Integrals_for_ROC_plots/8Hz_' + str(match_window_s) + 's_' + str(spike_position_jitter_std) + 's_jitter'

def argument_function(i):
    return (match_window, optimum_offset, i, spike_position_jitter_std)

def get_g_dynamics(match_window, optimum_offset, special_spine, spike_position_jitter_std):

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

    post_synaptic_stimulation_times += np.random.normal(loc=0, scale=spike_position_jitter_std, size=post_synaptic_stimulation_times.shape)

    model.axosomatic_compartments.add_iclamps(post_synaptic_stimulation_times, 1, 4)
    model.add_presynaptic_stimulation(presynaptic_stimulation_times)

    t = h.Vector().record(h._ref_t)
    h.finitialize(-75)
    h.continuerun(simulation_time_ms + 150)


    show_plot = False
    if show_plot:
        t = np.array(t)
        v = np.array(model.v_spine[special_spine + 5])
        g = np.array(model.g_spine[special_spine + 5])
        ca = np.array(model.ca_spine[special_spine + 5])
        fig, axes = plt.subplots(6, 1)

        for i in range(presynaptic_stimulation_times_special_spine.shape[0]):
            axes[0].axvline(presynaptic_stimulation_times_special_spine[i])

        for i in range(presynaptic_stimulation_times_special_spine.shape[0]):
            axes[1].axvline(presynaptic_stimulation_times_special_spine[i] + optimum_offset)

        for i in range(post_synaptic_stimulation_times.shape[0]):
            axes[2].axvline(post_synaptic_stimulation_times[i])

        axes[3].plot(t, v)
        axes[4].plot(t, ca)
        axes[4].axhline(1.5, color=prettyplot.colors['red'])
        axes[5].plot(t, g)
        for i in range(len(axes)):
            prettyplot.x_axis_only(axes[i])
            axes[i].set_xlim([t[0], t[-1]])
        plt.show()

    gs = np.array([g[-1] for g in model.g_spine])

    return {'gs': gs, 'special_spine': special_spine}



if __name__ == '__main__':

    parallelrun = ParallelRun(
        target_function=get_g_dynamics,
        argument_function=argument_function,
        max_number_of_processes=max_number_of_processes,
        iterations=len(special_spines),
    )

    outputs = parallelrun.run()

    gs = []
    spines = []

    for output in outputs:
        gs.append(output['gs'])
        spines.append(output['special_spine'])

    gs = np.array(gs)
    spines = np.array(spines)

    I = np.argsort(spines)
    gs = gs[I, :]
    spines = spines[I]

    np.savez(path, gs=gs, spines=spines, match_window=match_window, optimum_offset=optimum_offset)