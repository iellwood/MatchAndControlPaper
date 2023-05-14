"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

This script simplifies and smooths the complex geometry of the apical dendrite from
Hay et. al. PLoS Computational Biology, 2011, by fitting an exponential curve to the
apical dendrite radius as a function of distance fromthe soma.
"""


import numpy as np
import torch
import matplotlib.pyplot as plt


distances = [23.067805184264195, 38.44634197377366, 56.985559682117945, 78.68545830929705,
             100.38535693647617, 125.67107819383484, 146.98787721015685, 160.74993135526256, 174.51198550036827,
             185.11096666463686, 199.59568829830448, 221.12922338220824, 242.662758466112, 261.0883219287762,
             276.4059137702009, 291.7235056116256, 306.9881950262975, 325.7488131054729, 348.05826227590484,
             370.3677114463368, 390.95570979101336, 409.8222573099346, 428.6888048288559, 447.55535234777716,
             466.4218998666984, 485.2884473856197, 504.154994904541, 523.0215424234623, 541.8880899423835,
             560.7546374613048, 579.6211849802261, 598.4877324991473, 617.3542800180687]
diams = [7.20486823863129, 4.08698291062178, 3.6502690127734483, 3.2584230635936025,
         3.089999914169311,
         3.089999914169311, 3.089999914169311, 3.089999914169311, 3.089999914169311, 3.08999991416931,
         3.089999914169311, 2.569999933242798, 2.569999933242798, 2.5962909540493606, 2.5699999332427974,
         2.451774938921602, 2.6864140514981663, 3.089999914169312, 3.055301743259446, 2.494606981761777,
         2.5699999332427974, 2.31302307025813, 2.5699999332427983, 2.569999933242797, 2.5699999332427987,
         2.569999933242797, 2.569999933242797, 2.529957657700846, 2.4821465756433225, 2.8299999237060534,
         2.8299999237060534, 2.8299999237060534, 2.829999923706058]

distances = np.array(distances)
diams = np.array(diams)


diams = torch.tensor(diams, dtype=torch.float32)
distances = torch.tensor(distances, dtype=torch.float32)

A = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32))
B = torch.nn.Parameter(torch.tensor(2, dtype=torch.float32))
tau = torch.nn.Parameter(torch.tensor(10, dtype=torch.float32))

softplus = torch.nn.Softplus()

def y_model(x):
    return A + B * softplus(-x/tau)

def loss_function(x, y):
    return torch.mean(torch.square(y_model(x) - y))

optim = torch.optim.Adam([A, B, tau], 0.01)

for i in range(10000):

    loss = loss_function(distances, diams)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if i % 100 == 0:
        A_eval = A.detach().numpy()
        B_eval = B.detach().numpy()
        tau_eval = tau.detach().numpy()

        print('step', i, 'loss =', loss.detach().numpy(), A_eval, B_eval, tau_eval)

y_model_eval = y_model(distances).detach().numpy()

params = [2.7642336, 21.55532, 15.393625]

def softplus_numpy(x):
    return np.where(x > 0, x + np.log(1 + np.exp(-x)), np.log(1 + np.exp(x)))

d = distances.detach().numpy()
plt.plot(d, params[0] + params[1]*softplus_numpy(-d/params[2]))

plt.plot(distances.detach().numpy(), y_model_eval)
plt.plot(distances.detach().numpy(), diams.detach().numpy())
plt.show()


