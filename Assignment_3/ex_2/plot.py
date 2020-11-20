import matplotlib.pyplot as plt
import subprocess
import numpy as np
import pickle

GPUs = []
N = [10**i for i in range(1, 8)]
T = 100
b = 64

for n in N:
    output = subprocess.check_output(f"nvprof ./exercise_2b {n} {T} {b}",
                                     shell=True).decode().splitlines()
    GPU = int(output[0].split()[1].strip())

    print(f"GPU {b}", GPU)

    # Only add to CPUs once

    GPUs.append(GPU)

with open("managed-memory-rw.pkl", "wb") as pkl:
    pickle.dump({"x": N, "y": GPUs, "name": "managed-rw"}, pkl)

with open("unpinned-memory.pkl", "rb") as pkl:
    unpinned_pkl = pickle.load(pkl)

with open("pinned-memory.pkl", "rb") as pkl:
    pinned_pkl = pickle.load(pkl)

with open("managed-memory.pkl", "rb") as pkl:
    managed_pkl = pickle.load(pkl)

plt.figure(figsize=(10, 6))
plt.plot(managed_pkl["x"], managed_pkl["y"], label=managed_pkl["name"])
plt.plot(unpinned_pkl["x"], unpinned_pkl["y"], label=unpinned_pkl["name"])
plt.plot(pinned_pkl["x"], pinned_pkl["y"], label=pinned_pkl["name"])
plt.plot(N, GPUs, label="managed-with-(r|w)")
plt.legend(fontsize=16)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("array size", fontsize=16)
plt.ylabel("microseconds", fontsize=16)
plt.savefig("simulation-time.png")
