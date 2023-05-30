"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

Produces Figure 4A
"""

import matplotlib.pyplot as plt
import numpy as np
import prettyplot


def sigmoid(x):
    return np.where(x > 0, 1/(1 + np.exp(-x)), np.exp(x)/(1 + np.exp(x)))


def alpha(g, sharpness=20, threshold=1):
    return sigmoid(sharpness*(g/threshold - 1))


def threshold_function(g):
    a = alpha(g)
    return (1 - a) * 1 + a*8


xs = np.linspace(0, 2, 1000)
fig = prettyplot.figure_with_specified_size((6, 4), (1, 0.5), (2, 2))
prettyplot.xlabel('fraction of threshold')
prettyplot.ylabel('relative AMPA conductance')
plt.plot(xs, threshold_function(xs))
plt.axvline(1, color='k', linewidth=2)
plt.xticks([0, 0.5, 1, 1.5])
prettyplot.no_box()
plt.xlim(0, 1.5)
plt.savefig('../Figures/Figure_4A_ThresholdSigmoid.pdf', transparent=True)
plt.show()

