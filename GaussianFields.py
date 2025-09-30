# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: multipersEnv
#     language: python
#     name: python3
# ---

# %%
import powerbox as pbox
import torch
import multipers as mp
from multipers.filtrations import Cubical
from multipers.torch.diff_grids import get_grid, evaluate_mod_in_grid
import numpy as np
from utils import compute_mma_descriptor, fisher, simulate_grfs
import matplotlib.pyplot as plt

# %%
physical = False
N = 32
A, B = 1., 2.
theta = [A, B]
deltaA, deltaB = 0.1, 0.1
delta_theta = np.array([deltaA, deltaB])


pb = pbox.PowerBox(
    N = N,                     # Number of grid-points in the box
    dim = 2,                     # 2D box
    pk = lambda k: A*k**(-B), # The power-spectrum
    boxlength = N,           # Size of the box (sets the units of k in pk)
    vol_normalised_power=True, 
    ensure_physical = physical,
    seed = 42 # Random seed for reproducibility
)

TFM = fisher(theta, pb.k(), N)
print(f"The theoretical Fisher Information Matrix (TFM) is \n {TFM}")

# %% [markdown]
# ## MMA of a Gaussian Random Field

# %%
field = pb.delta_x()
gradient = compute_gradient(field, pb.x,pb.x)
result, mma = compute_mma_descriptor(field, gradient)
mma.plot(0)
# plt.show()
# mma.plot(1)
# plt.show()
# print(result)

# %% [markdown]
# ## Persistent Homology of GRFs

# %%
# field = torch.rand(10, 10, requires_grad=True)
# derivative = torch.rand(10, 10, requires_grad=True)
from gudhi import PeriodicCubicalComplex as PCC

n_s = 10000; n_d = 10000; dim_list = [0, 1]; 
seed = np.random.randint(1e4); print("seed = ", seed)

cl = PCC(period);
thetaLis = [[A + deltaA, B], [A, B + deltaB]]

ptsCov = simulate(theta, n_s)
pts = simulate(theta, n_d, seed = seed)
pts_pert = [simulate(thet, n_d, seed) for thet in thetaLis]

# %%
