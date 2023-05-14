"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

This script collects the data for Figure 4C.
"""

import numpy as np
import HocPythonTools
from Models.completeneuron import Neuron
from helper_functions import generate_spike_train_array, generate_spike_train
from parallel_run import ParallelRun


path = '../SimulationData/threshold_scan.npz'

optimum_offset = 7
match_window_size = 1000
number_of_postsynaptic_spikes = int(np.round(match_window_size * 0.008))
control_window_size = 1000
gap_window_size = 500

thresholds = {
    0.5:    0.05297869411632677,
    1:      0.07983869416080833,
    2:      0.12265248791940773
}

threshold = thresholds[1]

def run_simulation(threshold):

    x = np.load('../SimulationData/time_delays.npz')
    time_delays = x['time_delays']
    seed = 0
    np.random.seed(seed)

    # Set up the hoc interpreter and load the mechanisms
    h = HocPythonTools.setup_neuron('/home/iellwood/PycharmProjects/PyramidalNeuronModel/ChannelModFiles/x86_64/.libs/libnrnmech.so')
    model = Neuron(h, include_ca_dependent_potentiation=True)

    for branch in model.spiny_branches:
        for ampar in branch.ampa_list:
            ampar.ca_potentiation_threshold = threshold
            ampar.sigmoid_sharpness = 20
            ampar.tau_g_dynamic_delayed = 500

    # Parameters of simulation:
    number_of_spines = model.number_of_spines
    synapses_per_axon = 10
    special_spine = (np.random.randint(0, number_of_spines) // 10) * 10

    spike_rate_kHz = 0.008
    interspike_refactory_period_ms = 50

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

    scale = 0.08
    model.set_AMPA_weight('all', 0.1 * scale)  # 0.1 for 50 synapses
    model.set_NMDA_weight('all', 0.00005 * scale)  # 0.002 for 50 synapses

    # Run the simulation

    t = h.Vector().record(h._ref_t)

    h.finitialize(-75)
    h.continuerun(250 + match_window_size + gap_window_size + control_window_size + 250)

    t = np.array(t)
    v_spine = np.array(model.v_spine)[special_spine, :]
    ca_spine = np.array(model.ca_spine)[special_spine, :]
    g_spine = np.array(model.g_spine)[special_spine, :]
    v_soma = np.array(model.v_soma)


    output = {
        't': t,
        'v_spine': v_spine,
        'ca_spine': ca_spine,
        'g_spine': g_spine,
        'v_soma': v_soma,
        'postsynaptic_stimulation_times': postsynaptic_stimulation_times,
        'presynaptic_stimulation_times': presynaptic_stimulation_times,
        'special_spine': special_spine,
        'threshold': threshold,
        'seed': seed,
    }

    return output

threshold_multipliers = np.flip(np.geomspace(0.25, 4, 9))
print('threshold_multipliers =', np.round(threshold_multipliers, 3))


def argument_function(i):
    return (thresholds[match_window_size/1000] * threshold_multipliers[i], )


if __name__ == '__main__':
    parallelrun = ParallelRun(
        target_function=run_simulation,
        argument_function=argument_function,
        max_number_of_processes=12,
        iterations=len(threshold_multipliers),
    )

    outputs = parallelrun.run()

    vs = []
    ts = []
    thresholds = []
    for output in outputs:
        vs.append(output['v_soma'])
        ts.append(output['t'])
        thresholds.append(output['threshold'])

    vs = np.array(vs)
    ts = np.array(ts)
    thresholds = np.array(thresholds)

    I = np.argsort(thresholds)

    thresholds = thresholds[I]
    vs = vs[I, :]
    ts = ts[I, :]

    special_spine = outputs[0]['special_spine']
    presynaptic_stimulation_times = outputs[0]['presynaptic_stimulation_times'][special_spine]
    postsynaptic_stimulation_times = outputs[0]['postsynaptic_stimulation_times']

    np.savez(path, ts=ts, vs=vs, special_spine=special_spine, presynaptic_stimulation_times=presynaptic_stimulation_times, postsynaptic_stimulation_times=postsynaptic_stimulation_times)


