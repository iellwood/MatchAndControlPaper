"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

Produces Figure 4C
"""

import numpy as np
import matplotlib.pyplot as plt
import prettyplot

file_save_name = '../SimulationData/threshold_scan.npz'
data = np.load(file_save_name)
threshold_multipliers = np.flip(np.geomspace(0.25, 4, 9))

ts = data['ts']
vs = data['vs']
postsynaptic_stimulation_times = data['postsynaptic_stimulation_times']
presynaptic_stimulation_times = data['presynaptic_stimulation_times']

I = np.arange(vs.shape[0])
I = np.flip(I)
ts = ts[I, :] - 250
vs = vs[I, :]

fig, axes = plt.subplots(ts.shape[0] + 1 - 3, 1, figsize=(4, 2))
for i in range(1, len(axes)):
    axes[i].plot(ts[i + 2, :], vs[i + 2, :])
    threshold = threshold_multipliers[i + 2]
    print(threshold)
    axes[i].text(ts[0, 0] - 100, 0.3, str(np.round(threshold, 2)), rotation=0, transform=axes[i].get_xaxis_text1_transform(0)[0])


for i in range(len(presynaptic_stimulation_times)):
    axes[0].axvline(presynaptic_stimulation_times[i] - 250, color=prettyplot.colors['red'])

for i in range(len(postsynaptic_stimulation_times)):
    axes[0].axvline(postsynaptic_stimulation_times[i] - 250, color='k')

for i in range(len(axes)):
    axes[i].set_xlim([ts[0, 0], ts[0, -1]])

for i in range(len(axes) - 1):
    axes[i].axis('off')

prettyplot.x_axis_only(axes[-1])
plt.savefig('../Figures/Figure_4C_ThresholdScan.pdf', transparent=True)

plt.show()


