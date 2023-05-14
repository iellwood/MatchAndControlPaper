"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

Produces Figure 3A
"""

import matplotlib.pyplot as plt
import numpy as np
import prettyplot
import os

x = np.load('../SimulationData/time_delays.npz')
time_delays = x['time_delays']


def find_maximum_with_second_order_approximation(times, x):
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

path = '../SimulationData/Ca_integral_at_different_offsets_per_spine/offset_data.npz'

data = np.load(path)

g_dynamic_outputs = data['gs']
print('dataset size =', g_dynamic_outputs.shape)

spine_numbers = data['spine_numbers']
offset_times = np.linspace(-5, 20, 26)

fs = 1/(offset_times[1] - offset_times[0])

import scipy.signal as signal
sos = signal.butter(4, [0.2], btype='low', output='sos', fs=fs)
g_dynamic_outputs_filtered = signal.sosfiltfilt(sos, np.mean(g_dynamic_outputs, 2), axis=1)

g_dynamic_outputs_filtered_offset = g_dynamic_outputs_filtered*0

for i in range(g_dynamic_outputs_filtered.shape[0]):
    g_dynamic_outputs_filtered_offset[i, :] = np.interp(offset_times, offset_times + time_delays[spine_numbers[i]], g_dynamic_outputs_filtered[i, :])


prettyplot.figure_with_specified_size((5, 4), (1, 0.5), (1.5, 1.5))


prettyplot.plot_with_sem(offset_times, np.transpose(g_dynamic_outputs_filtered_offset, axes=[1, 0]), color=prettyplot.colors['blue'], fillcolor=[0.8, 0.8, 1.0])
prettyplot.no_box()
prettyplot.xlabel('delta t')
plt.yticks([], [])
prettyplot.ylabel('[Ca]^4 integral')
plt.ylim([0, 0.028])
plt.xlim([-5, 20])
max_t = find_maximum_with_second_order_approximation(offset_times, np.mean(g_dynamic_outputs_filtered_offset, 0))
print('best offset =', max_t)
#plt.axvline(max_t, color='k', linewidth=2, linestyle='--')

#plt.gca().text(max_t + 0.2, 0.5, str(np.round(max_t, 2)), rotation=90, transform=plt.gca().get_xaxis_text1_transform(0)[0])
plt.savefig('../Figures/CaVsOffset.pdf')
prettyplot.title('average over 300 runs')
plt.show()
