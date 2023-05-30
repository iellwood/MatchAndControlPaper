import numpy as np
import matplotlib.pyplot as plt

def collect_fraction_arrays(match_window_size_s):

    data = np.load('../SimulationData/ControlPhaseTests/8Hz_' + str(match_window_size_s) + 's.npz', allow_pickle=True)

    presynaptic_stimulation_times = data['presynaptic_stimulation_times']
    somatic_spike_times = data['somatic_spike_times']
    postsynaptic_stimulation_times = data['postsynaptic_stimulation_times']
    special_spines = data['special_spines']

    t_0 = 250 + data['match_window'] + 500

    fraction_spurious_spikes = np.zeros((len(postsynaptic_stimulation_times),))
    fraction_failed_spikes = np.zeros((len(postsynaptic_stimulation_times),))
    number_of_stims = np.zeros((len(postsynaptic_stimulation_times),))

    for i in range(len(postsynaptic_stimulation_times)):
        number_of_stims[i] = len(postsynaptic_stimulation_times[i])

    for i in range(len(presynaptic_stimulation_times)):
        pres = presynaptic_stimulation_times[i][special_spines[i]]
        pres = pres[pres >= t_0]
        soms = somatic_spike_times[i]
        soms = soms[soms > t_0]



        false_spikes = 0
        for j in range(len(soms)):
            if np.min(np.abs(soms[j] - (pres + 10))) > 10:
                false_spikes += 1
        if len(soms) > 0:
            fraction_spurious_spikes[i] = false_spikes / len(soms)
        else:
            fraction_spurious_spikes[i] = 0

        failed_spikes = 0
        if len(soms) > 0:
            for j in range(len(pres)):
                if np.min(np.abs(pres[j] - (soms - 10))) > 10:
                    failed_spikes += 1
        else:
            if len(pres) > 0:
                failed_spikes = len(pres)
        if len(pres) > 0:
            fraction_failed_spikes[i] = failed_spikes / len(pres)
        else:
            fraction_failed_spikes[i] = 0

    return fraction_failed_spikes, fraction_spurious_spikes, number_of_stims


fraction_failed_spikes = []
fraction_spurious_spikes = []
number_of_stims = []

match_windows = [0.5, 1, 2]

for i in range(len(match_windows)):
    print('Collecting data from match window =', match_windows[i])
    ffs, fss, nos = collect_fraction_arrays(match_windows[i])
    fraction_failed_spikes.append(ffs)
    fraction_spurious_spikes.append(fss)
    number_of_stims.append(nos)

fraction_failed_spikes = np.array(fraction_failed_spikes)
fraction_spurious_spikes = np.array(fraction_spurious_spikes)
number_of_stims = np.array(number_of_stims)

np.savez('../SimulationData/ControlPhaseTests/failed_and_spurious_spikes',
         fraction_failed_spikes=fraction_failed_spikes,
         fraction_spurious_spikes=fraction_spurious_spikes,
         number_of_stims=number_of_stims,
         match_windows=match_windows)