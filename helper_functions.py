"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

Misc. helper functions used throughout the project
"""

import numpy as np

def generate_spike_train(time_range, spike_rate, recovery_time):
    """
    Generates a spike train drawn from a Poisson process with fixed time_range, spike_rate and recovery_time

    :param time_range: Should be a list [t_0, t_1]
    :param spike_rate: The spike-rate
    :param recovery_time: The minimum time separation between spikes
    :return: The spike train as a np.array
    """
    current_t = time_range[0]
    beta = 1/spike_rate
    spike_times = []

    while current_t < time_range[1]:

        t_new_spike = np.random.exponential(scale=beta) + current_t
        current_t = t_new_spike + recovery_time

        if t_new_spike <= time_range[1]:
            spike_times.append(t_new_spike)

    return np.array(spike_times)


def generate_spike_train_array(count, time_range, spike_rate, recovery_time, synapses_per_axon=1, multi_synapse_arangement='neighbors'):
    """
    Generates a list of spike trains using generate_spike_train.
    Note that synapses_per_axon specifies how many of the spike trains will be equal to each other.
    For example, if synapses_per_axon = 10, every spike train will be repeated 10 times.

    :param count: Number of spike trains to generate
    :param time_range: A list representing the start and finish of the spike trains [t_0, t_1]
    :param spike_rate: The spike-rate
    :param recovery_time: The minimum time between spikes
    :param synapses_per_axon: See note above
    :param multi_synapse_arangement: Must be 'neighbors' or 'random'. If 'neighbors', all the repeated
    spike trains will be adjacent to each other in the list of spike trains.
    :return: A list of spike-trains
    """
    trains = []
    spike_train = None
    for i in range(count):
        if i % synapses_per_axon == 0:
            spike_train = generate_spike_train(time_range, spike_rate, recovery_time)
        trains.append(spike_train)
    if multi_synapse_arangement == 'neighbors' or multi_synapse_arangement is None:
        pass
    elif multi_synapse_arangement == 'random':
        trains = list(np.random.permutation(np.array(trains, dtype=object)))
    else:
        raise Exception('Unknown multi-synapse arrangement.')

    return trains


def make_vector_from_ts(ts, ts_range, bin_size):
    """
    Vectorizes a spike-train

    :param ts: List of spike-times
    :param ts_range: window size for possible spike times
    :param bin_size: bin size for pooling the spikes.
    :return: vectorized spike train
    """
    dt = ts_range[-1] - ts_range[0]
    n = int(np.round(dt / bin_size))

    v = np.zeros([n])

    s = 1 / (ts_range[1] - ts_range[0])

    for i in range(len(ts)):
        t_between_0_and_1 = (ts[i] - ts_range[0]) * s
        t_index_with_fraction = np.minimum(np.maximum(t_between_0_and_1, 0), 1) * (n - 1)

        index = int(np.floor(t_index_with_fraction))

        fraction = t_index_with_fraction - index

        if index < n - 1:
            v[index] += (1 - fraction)
            v[index + 1] += fraction
        else:
            v[index] += 1

    return v


def decimate_trace(ts, x, skip, axis=-1):
    """
    Smoothes and then decimates a signal.

    :param ts: times of signal
    :param x: signal
    :param skip: how many datapoints to skip
    :param axis: axis to perform the operation
    :return: decimated signal
    """
    fs = 1/(ts[1] - ts[0])
    nyquist_frequency = 1/(ts[skip] - ts[0]) / 2
    import scipy.signal as signal
    sos = signal.butter(4, nyquist_frequency, btype='low', fs=fs, output='sos')
    x_filtered = signal.sosfiltfilt(sos, x, axis=axis)
    I = np.arange(0, x_filtered.shape[axis], step=skip)

    return np.take(x_filtered, indices=I, axis=axis)

def find_maximum_with_second_order_approximation(times, x):
    """
    Uses a second order polynomial to find the time when the
    approximate peak of a discrete signal occurs.

    :param times: The time values associated with the signal x
    :param x: The signal
    :return: The time where the maximum of x occurs.
    """
    i = np.argmax(x)
    if i == 0:
        return times[0]
    elif i == len(x) - 1:
        return times[-1]
    else:
        coeffs = np.polyfit([-1, 0, 1], [x[i - 1], x[i], x[i + 1]], 2)
        alpha = -coeffs[1]/(2 * coeffs[0])
        i_low = int(np.floor(i + alpha))
        fractional_part = (i + alpha) % 1
        return (1 - fractional_part)*times[i_low] + fractional_part*(times[i_low + 1])