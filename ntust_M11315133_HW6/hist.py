#!/usr/bin/env python3

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

grid = np.power(2, np.arange(5, 11))

if not os.path.isdir("results"):
    os.mkdir("results")

for g in grid:
    if os.path.isfile(f"results/out_gmem_{g}.txt"):
        continue
    get_ipython().system('./hist_1gpu_gmem <<< "0 81920000 0 20 33 {g} 128 1" >> results/out_gmem_{g}.txt')

files = glob.glob("results/out_gmem_*.txt")
files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
results = []
for f in files:
    with open(f, "r") as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Speed"):
                results.append(float(line.split("=")[-1].strip()))

plt.plot(grid, results, linestyle="-")

mean = np.mean(results)
plt.axhline(mean, color="green", linestyle="--", label="Mean")
plt.annotate(
    f"Mean: {mean:.3f}",
    (grid[0], mean),
    textcoords="offset points",
    xytext=(340, 10),
    ha="center",
)
plt.title("Performance of histogramming 1 GPU with global memory")
plt.xlabel("Block size")
plt.ylabel("Speed up rate")
plt.savefig("results/gmem.png")


get_ipython().system('./hist_1gpu_shmem <<< "0 81920000 0 20 33 33 128 1" >> results/out_shmem.txt')

data_gpu = np.loadtxt("hist_gmem.txt")
data_gpu_shmem = np.loadtxt("hist_shmem.txt")
data_cpu = np.loadtxt("hist_cpu.txt")

print(data_gpu.T[0])
print(data_gpu.T[1])

plt.figure(figsize=(8, 3), dpi=450)
plt.subplot(1, 3, 1)
plt.title("GPU result")
plt.bar(data_gpu.T[0], data_gpu.T[1] / 81920000, label="54.53 (ms)")
x = np.linspace(0, 20, 32)
plt.plot(x, np.exp(-x) / np.sum(np.exp(-x)), color="r", label="exp(-x)")
plt.ylabel("P(x)")
plt.xlabel("x")
plt.legend()
plt.xlim(0, 10)

plt.subplot(1, 3, 2)
plt.title("GPU shmem result")
plt.bar(data_gpu_shmem.T[0], data_gpu_shmem.T[1] / 81920000, label="25.14 (ms)")
x = np.linspace(0, 20, 32)
plt.plot(x, np.exp(-x) / np.sum(np.exp(-x)), color="r", label="exp(-x)")
plt.legend()
plt.xlabel("x")
plt.xlim(0, 10)


plt.subplot(1, 3, 3)
plt.title("CPU result")
plt.bar(data_cpu.T[0], data_cpu.T[1] / 81920000, label="55.00 (ms)")
x = np.linspace(0, 20, 32)
plt.plot(x, np.exp(-x) / np.sum(np.exp(-x)), color="r", label="exp(-x)")
plt.legend()
plt.xlabel("x")
plt.xlim(0, 10)

plt.savefig("results/histogram_results.png")
