#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./result/result.csv")

df["threads_per_block"] = pd.to_numeric(df["threads_per_block"], errors='coerce')
df["gflops"] = pd.to_numeric(df["gflops"], errors='coerce')
df = df.dropna(subset=["threads_per_block", "gflops"])
df = df.sort_values("threads_per_block")

x = df["threads_per_block"].to_numpy()
y = df["gflops"].to_numpy()

plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-')
plt.title("GPU Performance vs Threads Per Block")
plt.xlabel("Threads Per Block")
plt.ylabel("GFLOPS")
plt.grid(True)
plt.savefig("./result/result_graph.png", dpi=300)
plt.show()

