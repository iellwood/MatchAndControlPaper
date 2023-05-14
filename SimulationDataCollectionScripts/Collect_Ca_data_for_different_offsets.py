"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

This script collects the data for Figure 3A.
"""

import numpy as np
import pickle
import HocPythonTools
import time
import datetime
import multiprocessing
from multiprocessing import Queue
import os
import prettyplot

offset_times = np.linspace(-5, 20, 26)
repetitions = 10
max_spine = 1500
spine_skip = 50

path = '../SimulationData/Ca_integral_at_different_offsets_per_spine/offset_data'

def compute_offsets_process(queue, spine_number, times, repetitions):
    import matplotlib.pyplot as plt
    np.random.seed((os.getpid() * int(time.time()*10000)) % 123456789)
    t_0 = time.time()
    from helper_functions import generate_spike_train_array, generate_spike_train
    from Models.completeneuron import Neuron


    # Initialize the hoc interpreter and load all mechanisms
    # Set up the hoc interpreter and load the mechanisms
    h = HocPythonTools.setup_neuron('/home/iellwood/PycharmProjects/PyramidalNeuronModel/ChannelModFiles/x86_64/.libs/libnrnmech.so')
    model = Neuron(h, include_ca_dependent_potentiation=False)

    distances = []
    for i in range(model.number_of_spines):
        branch_number = model.branch_ID[i]
        branch_spine_number = model.branch_spine_number[i]

        distances.append(h.distance(model.spiny_branches[branch_number].spine_list[branch_spine_number].head(0.5)))
    distances = np.array(distances)

    # Parameters of simulation:
    number_of_spines = model.number_of_spines
    synapses_per_axon = 10

    pid = os.getpid()

    simulation_time_ms = 250
    presynaptic_spike_rate_kHz = 0.008
    interspike_refactory_period_ms = 50

    def perform_run(special_spine, t_offset=5):
        presynaptic_stimulation_times = generate_spike_train_array(
            number_of_spines,
            [0, simulation_time_ms + 100],
            presynaptic_spike_rate_kHz,
            recovery_time=interspike_refactory_period_ms,
            synapses_per_axon=synapses_per_axon,
            multi_synapse_arangement='neighbors'
        )

        found_non_empty_spike_train = False
        while not found_non_empty_spike_train:
            presynaptic_stimulation_times_special_spine = generate_spike_train(
                [50, simulation_time_ms + 50],
                0.008,
                recovery_time=50,
            )
            if presynaptic_stimulation_times_special_spine.shape[0] == 2:
                found_non_empty_spike_train = True

        for i in range(10):
            presynaptic_stimulation_times[special_spine + i] = presynaptic_stimulation_times_special_spine

        psts = presynaptic_stimulation_times[special_spine]
        post_synaptic_stimulation_times = psts + t_offset

        model.axosomatic_compartments.add_iclamps(post_synaptic_stimulation_times, 1, 4)
        model.add_presynaptic_stimulation(presynaptic_stimulation_times)

        t = h.Vector().record(h._ref_t)
        h.finitialize(-75)
        h.continuerun(simulation_time_ms + 100)


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
                axes[1].axvline(presynaptic_stimulation_times_special_spine[i] + t_offset)

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


        final_g_dynamic = model.g_spine[special_spine][-1]
        return final_g_dynamic

    g_dynamic_outputs = np.zeros(shape=(len(times), repetitions))

    for repetition_number in range(repetitions):
        for time_index in range(len(times)):
            g_dynamic_outputs[time_index, repetition_number] = perform_run(spine_number, times[time_index])


    queue.put({'spine': spine_number, 'g': g_dynamic_outputs})


def get_from_queue(queue, outputs, t_0):
    if not queue.empty():
        outputs.append(queue.get())

        delta_t = time.time() - t_0
        time_per_trial = delta_t / len(outputs)
        time_remaining = (max_spine // spine_skip - len(outputs)) * time_per_trial
        print('Computed =', len(outputs), ' spines of', max_spine // 10,
              'Time elapsed =', str(datetime.timedelta(seconds=delta_t)),
              'time_remaining =', str(datetime.timedelta(seconds=time_remaining))
              )

if __name__ == '__main__':

    queue = Queue()

    t_0 = time.time()

    print('offset_times =', offset_times)

    spine_number = 0

    processes = []

    outputs = []

    while spine_number < max_spine:

        processes = [process for process in processes if process.is_alive()]

        if len(processes) < 10:
            process = multiprocessing.Process(
                target=compute_offsets_process,
                args=(queue, spine_number, offset_times, repetitions)
            )
            process.start()
            processes.append(process)
            spine_number += spine_skip

        get_from_queue(queue, outputs, t_0)

        time.sleep(0.5)

    while len(processes) > 0:
        processes = [process for process in processes if process.is_alive()]
        get_from_queue(queue, outputs, t_0)
        time.sleep(1)

    while not queue.empty():
        get_from_queue(queue, outputs, t_0)

    spines = []
    tensors = []

    for o in outputs:
        spines.append(o['spine'])
        tensors.append(o['g'])

    spines = np.array(spines)
    tensors = np.array(tensors)

    I = np.argsort(spines)

    spines = spines[I]
    print('spines =', spines)
    print('tensors.shape =', tensors.shape)
    tensors = tensors[I, :, :]


    np.savez(path, spine_numbers=spines, gs=tensors, times=offset_times)


