#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)


# In[2]:


def w(x, c, a):
    """
    Compute the weight function w(x) = a * exp(-c * x^2).

    Parameters:
    x (float or np.ndarray): Input value(s).
    c (float): Scaling factor.
    a (float): Decay factor.

    Returns:
    float or np.ndarray: Computed weight(s).
    """
    return c * np.exp(-a * x)


# In[3]:


xs = [np.random.uniform(0, 1, 1000) for _ in range(10)]
ws = [w(x, 1, 1) for x in xs]
np.average(np.prod(ws, axis=0))


# In[4]:


def find_norm(n):
    xs = [np.random.uniform(0, 1, n) for _ in range(10)]
    ws = [w(x, 1, 1) for x in xs]
    return 1/np.power(np.average(np.prod(ws, axis=0)), 1/10)


# In[5]:


grid = [2**i for i in range(6, 17)]
nfs = []
for g in grid:
    nfs.append(nf := find_norm(g))
    print(f"Sampling number {g}: Normalization factor = {nf}")


# In[6]:


xs = [np.random.uniform(0, 1, 65536) for _ in range(10)]
ws = [w(x, 1.5822955636702438, 1) for x in xs]
np.mean(np.prod(ws, axis=0))


# In[7]:


for g in grid:
    if os.path.isfile(f"results/out_{g}.txt"):
        continue
    get_ipython().system('./mc_int_cpu <<< "{g}" >> results/out_{g}.txt')
    os.rename("results/mc_int_cpu.txt", f"results/mc_int_cpu_{g}.txt")


# In[8]:


fns = glob.glob("results/mc_int_cpu_*.txt")
fns = sorted(fns, key=lambda x: int(x.split("_")[-1].split(".")[0]))

simple_results_mean = []
simple_results_std = []
metro_results_mean = []
metro_results_std = []
for fn in fns:
    with open(fn, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Simple"):
                simple_results_mean.append(
                    float(line.split(":")[1].split("+")[0].strip())
                )
                simple_results_std.append(
                    float(line.split(":")[1].split("-")[1].strip())
                )
            if line.startswith("Metropolis"):
                metro_results_mean.append(
                    float(line.split(":")[1].split("+")[0].strip())
                )
                metro_results_std.append(
                    float(line.split(":")[1].split("-")[1].strip())
                )


# In[9]:


plt.plot(grid, simple_results_mean, label="Simple Sampling MC", marker="o")
plt.fill_between(
    grid,
    np.array(simple_results_mean) - np.array(simple_results_std),
    np.array(simple_results_mean) + np.array(simple_results_std),
    alpha=0.2
)
plt.plot(grid, metro_results_mean, label="Metropolis Importance Sampling MC", marker="o")
plt.fill_between(
    grid,
    np.array(metro_results_mean) - np.array(metro_results_std),
    np.array(metro_results_mean) + np.array(metro_results_std),
    alpha=0.2
)
plt.xlabel("Number of samples")
plt.ylabel("Integral value")
plt.xscale("log")
# plt.yscale("log")
plt.title("Monte Carlo Integration Results")
plt.legend()

