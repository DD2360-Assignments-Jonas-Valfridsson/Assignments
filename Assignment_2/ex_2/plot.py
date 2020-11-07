import matplotlib.pyplot as plt
import subprocess
import numpy as np



N = [10**i for i in range(1, 10)]
CPUs, GPUs = [], []
for n in N:
    output = subprocess.check_output(f"nvprof ./exercise_2 {n} 32", shell=True).decode().splitlines()
    CPU = int(output[0].split()[1].strip())
    GPU = int(output[1].split()[1].strip())

    print("CPU", CPU)
    print("GPU", GPU)
    
    CPUs.append(CPU)
    GPUs.append(GPU)

plt.figure(figsize=(10, 6))
plt.plot(N, CPUs, label="CPU")
plt.plot(N, GPUs, label="GPU")
plt.legend(fontsize=16)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("array size", fontsize=16)
plt.ylabel("microseconds", fontsize=16)
plt.savefig("saxpy-cpu-vs-gpu.png")
plt.show()
