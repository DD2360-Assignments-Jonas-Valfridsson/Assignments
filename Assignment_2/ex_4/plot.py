import matplotlib.pyplot as plt
import subprocess
import numpy as np

batch = [16, 32, 64, 128, 256]
N = [10**i for i in range(5, 9)]
CPUs, GPUs = [], {b: [] for b in batch}
CPUs_PI, GPUs_PI = [], {b: [] for b in batch}
samples_per_thread = 2000

for b in batch:
    for n in N:
        output = subprocess.check_output(
            f"nvprof ./exercise_4 {n} {samples_per_thread} {b}",
            shell=True).decode().splitlines()
        print("output", output)
        CPU_PI = float(output[0].split()[1].strip())

        CPU = int(output[1].split()[1].strip())
        GPU_PI = float(output[2].split()[1].strip())
        GPU = int(output[3].split()[1].strip())

        print("CPU", CPU)
        print(f"GPU {b}", GPU)

        # Only add to CPUs once
        if b == 16:
            CPUs.append(CPU)
            CPUs_PI.append(CPU_PI)

        GPUs[b].append(GPU)
        GPUs_PI[b].append(GPU_PI)

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.title("Performance")
plt.plot(N, CPUs, label="CPU")
for b, gpu in GPUs.items():
    plt.plot(N, gpu, label=f"GPU {b}")
plt.legend(fontsize=16)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("sample size", fontsize=16)
plt.ylabel("microseconds", fontsize=16)
plt.subplot(2, 1, 2)
plt.title("PI-Estimate")
plt.plot(N, CPUs_PI, label="CPU")
for b, gpu in GPUs_PI.items():
    plt.plot(N, gpu, label=f"GPU {b}")
plt.plot(N, np.ones(len(N)) * np.pi, label="True")
plt.xscale("log")
plt.xlabel("sample size", fontsize=16)
plt.ylabel("pi estiamte", fontsize=16)
plt.legend(fontsize=16)

plt.tight_layout()
plt.savefig("pi-float-cpu-vs-gpu.png", bbox_inches="tight")
plt.show()
