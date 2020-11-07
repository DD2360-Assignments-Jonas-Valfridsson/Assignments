import matplotlib.pyplot as plt
import subprocess
import numpy as np



batch = [16, 32, 64, 128, 256]
N = [10**i for i in range(1, 7)]
CPUs, GPUs = [], {b: [] for b in batch}
T = 1000


for b in batch:
    for n in N:
        output = subprocess.check_output(f"nvprof ./exercise_3 {n} {T} {b}", shell=True).decode().splitlines()
        CPU = int(output[0].split()[1].strip())
        GPU = int(output[1].split()[1].strip())

        print("CPU", CPU)
        print(f"GPU {b}", GPU)
        
        # Only add to CPUs once
        if b == 16:
            CPUs.append(CPU)

        GPUs[b].append(GPU)

plt.figure(figsize=(10, 6))
plt.plot(N, CPUs, label="CPU")
for b, gpu in GPUs.items():
    plt.plot(N, gpu, label=f"GPU {b}")
plt.legend(fontsize=16)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("array size", fontsize=16)
plt.ylabel("microseconds", fontsize=16)
plt.savefig("simulation-cpu-vs-gpu.png")
plt.show()
