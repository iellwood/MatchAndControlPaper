"""
I. T. Ellwood, Short-term Hebbian learning can implement transformer-like attention

This script produces the Figure 2 panels C & D, which show the optimum kernel (panel C)
and the quality of the fit (Figure 2D)
"""

import torch
import numpy as np
from helper_functions import make_vector_from_ts
import matplotlib.pyplot as plt
import prettyplot

torch.manual_seed(0)

q = np.load('../SimulationData/thresholds.npz', allow_pickle=True)
thresholds = q['thresholds']

# Used to normalize the calcium integrals to a fraction of the threshold
# thresholds = {
#     0.5:    0.05297869411632677,
#     1:      0.07983869416080833,
#     2:      0.12265248791940773
# }

match_window_s = 1
bin_size = 1
match_window = match_window_s * 1000
path = '../SimulationData/OverlapData/6Hz_' + str(match_window_s) + 's.npz'
path_test = '../SimulationData/OverlapData/TEST_DATA_6Hz_' + str(match_window_s) + 's.npz'

def load_data_into_torch_tensors(path):
    """
    Processes the data in path and returns torch tensors for the fitting procedure
    :param path: path to the data
    :return: presynaptic spike trains vectorized, postsynaptic spike trains vectorized, calcium**4 integrals
    """
    x = np.load('../SimulationData/time_delays.npz')
    time_delays = x['time_delays']
    time_delays = time_delays[::10]

    data = np.load(path, allow_pickle=True)


    gs = data['gs']
    pres = data['presynaptic_stimulation_times']
    posts = data['postsynaptic_stimulation_times']

    def every_tenth_spine(x):
        y = []
        for i in range(len(x)):
            y.append(x[i][::10])
        return y

    pres = every_tenth_spine(pres)
    gs = gs[:, ::10]

    pres_vectorized = []
    for pre in pres:
        vs = []
        for i in range(len(pre)):
            v = make_vector_from_ts(pre[i] - time_delays[i], [50, 50 + match_window], bin_size)
            vs.append(v)

        pres_vectorized.append(vs)

    pres_vectorized = np.array(pres_vectorized)

    posts_vectorized = []
    for post in posts:
        posts_vectorized.append(make_vector_from_ts(post, [50, 50 + match_window], bin_size))


    posts_vectorized = np.array(posts_vectorized)
    posts_vectorized = np.expand_dims(posts_vectorized, 1)
    posts_vectorized = posts_vectorized * np.ones(shape=[1, pres_vectorized.shape[1], 1])
    posts_vectorized = np.reshape(posts_vectorized, [-1, posts_vectorized.shape[2]])
    pres_vectorized = np.reshape(pres_vectorized, [-1, pres_vectorized.shape[2]])
    gs = np.reshape(gs, [-1])

    A = torch.tensor(pres_vectorized, dtype=torch.float32, requires_grad=False)
    B = torch.tensor(posts_vectorized, dtype=torch.float32, requires_grad=False)
    G = torch.tensor(gs, dtype=torch.float32, requires_grad=False)

    return A, B, G


A, B, G = load_data_into_torch_tensors(path)
G = G/thresholds[1]
A_test, B_test, G_test = load_data_into_torch_tensors(path_test)
G_test = G_test/thresholds[1]

# Remove outliers more than 4 std away from mean:
I = (G - torch.mean(G))/torch.std(G) < 4
A = A[I, :]
B = B[I, :]
G = G[I]
print('number_of_outliers (training data) =', I.shape[0] - torch.sum(I).detach().numpy(), 'out of', I.shape[0])

I_test = (G_test - torch.mean(G_test))/torch.std(G_test) < 4
A_test = A_test[I_test, :]
B_test = B_test[I_test, :]
G_test = G_test[I_test]
print('number_of_outliers (test data) =', I_test.shape[0] - torch.sum(I_test).detach().numpy())


# Kernel Parameter. This is the output of the fit
K = torch.zeros(size=(2 * B.shape[1] - 1,), dtype=torch.float32, requires_grad=True)

ts = torch.linspace(-match_window, match_window, K.shape[0])

# This is a window that suppresses the Kernel away from the origin. Note that, even if this
# window is not included, the kernel is small away from t = 0. However, this effectively
# reduces the number of parameters.
Z = torch.sigmoid((ts + 30)/10) * torch.sigmoid(-(ts - 30)/10)

# Function that multiplies the kernel by Z, the window
def k_nlin(x):
    return Z * x

# Creates a Toeplitz matrix out of the kernel. Multiplying a vector by the Toeplitz matrix
# is essentially the same as convolving the vector with the kernel
def Toeplitz(x):
    n = x.shape[0]
    m = (n + 1)//2
    rows = []
    for i in range(m):
        rows.append(x[m - i - 1: 2*m - i - 1, None])
    return torch.concat(rows, 1)

# model for the calcium**4 integral
def g_model(a, b, k):
    k = k_nlin(k)
    aK = torch.matmul(a, Toeplitz(k))
    return torch.sum(aK * b, 1)

# This adds a slight bias towards kernels that are smooth. Did not have a major benefit in the fit.
def K_smoothness_loss(k):
    k = k_nlin(k)
    return (torch.sum(torch.square(torch.diff(k, 1))) + torch.square(k[-1]) + torch.square(k[0]))

# Beginning of torch code to compute the fit by gradient descent.

adam_optimizer = torch.optim.Adam([K], lr=.0001)


def step(i):
    I = torch.randint(0, A.shape[0], size=(1000,))
    adam_optimizer.zero_grad()
    g_m = g_model(A[I], B[I], K)
    loss = torch.mean(torch.square(g_m - G[I])) + 0.01* K_smoothness_loss(K)
    loss.backward()
    adam_optimizer.step()

    if i % 100 == 0:
        g_m = g_model(A, B, K).cpu().detach().numpy()
        g_true = G.cpu().detach().numpy()
        ev = 1 - np.var(g_true - g_m) / np.var(g_true)

        g_m_test = g_model(A_test, B_test, K).cpu().detach().numpy()
        g_true_test = G_test.cpu().detach().numpy()
        ev_test = 1 - np.var(g_true_test - g_m_test) / np.var(g_true_test)

        print(i, 'explained_variance =', [np.round(ev, 3), np.round(ev_test, 3)])

for i in range(10000):
    adam_optimizer.lr = 200/(i + 200) # slightly decrease the learning rate over time
    step(i)

# Makes Figure 2 C
fig = prettyplot.figure_with_specified_size((8, 4), (1, 0.5), (4, 2))
prettyplot.title('Best fit overlap kernel')
ts = np.linspace(-match_window, match_window, K.shape[0])
plt.plot(ts, np.flip(k_nlin(K).cpu().detach().numpy()))
plt.xlim([-100, 100])
prettyplot.x_axis_only()
prettyplot.xlabel('t (ms)')
plt.savefig('../Figures/KernelModelOfOverlapPlots/Figure_2C_Kernel.pdf', transparent=True)
plt.show()

# Computes the explained variance of the model:
g_m = g_model(A, B, K).cpu().detach().numpy()
g_true = G.cpu().detach().numpy()
explained_variance = 1 - np.var(g_true - g_m) / np.var(g_true)

g_m_test = g_model(A_test, B_test, K).cpu().detach().numpy()
g_true_test = G_test.cpu().detach().numpy()
ev_test = 1 - np.var(g_true_test - g_m_test) / np.var(g_true_test)
print('explained variance training =', explained_variance)
print('explained variance test =', ev_test)

# Makes Figure 2 D
fig = prettyplot.figure_with_specified_size((5, 6), (1, 0.5), (2, 3))
I = torch.randint(0, A_test.shape[0], size=(1000,))
g_m_test = g_model(A_test[I, :], B_test[I, :], K).cpu().detach().numpy()
g_true_test = G_test[I].cpu().detach().numpy()
plt.scatter(g_m_test, g_true_test,  marker='.', color=prettyplot.colors['blue'])
plt.xlim([0, np.max(g_m_test)*1.05])
plt.ylim([0, np.max(g_true_test)*1.05])
plt.gca().set_aspect(1)
plt.axline([0, 0], slope=1, linewidth=2, color='k', linestyle='--')
prettyplot.xlabel('g model')
prettyplot.ylabel('g true')
prettyplot.no_box()
plt.text(np.max(g_m_test)*0.8, np.max(g_true_test)*0.8, 'R^2 = ' + str(np.round(ev_test, 2)))
prettyplot.title('match window = ' + str(match_window_s) + ' s')
plt.savefig('../Figures/KernelModelOfOverlapPlots/Figure_2D_FitQuality.pdf', transparent=True)
plt.show()


