import seaborn as sb
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import pickle
import pandas as pd

size = [64, 128, 256, 512, 1024, 2048, 4096]

CPU = []
GPUs = [[] for _ in range(3)]


for s in size:
    output = subprocess.check_output(f"./exercise_3 -s {s} -v",
                                     shell=True).decode().splitlines()
    cpu = float(output[0].strip()) 
    naive = float(output[1].strip()) 
    shmem = float(output[2].strip()) 
    cublas = float(output[3].strip()) 
    
    CPU.append(cpu)
    GPUs[0].append(naive)
    GPUs[1].append(shmem)
    GPUs[2].append(cublas)

    print(f"s {s} -- CPU {cpu} - naive {naive} - shmem {shmem} - cublas {cublas}")

plt.figure(figsize=(10, 6))

df = pd.DataFrame({
    "name": ((["CPU"] * len(size)) 
        + (["naive"] * len(size))
        + (["shmem"] * len(size))
        + (["cublas"] * len(size))
        ),
    "size": size * 4,
    "time": CPU + GPUs[0] + GPUs[1] + GPUs[2] 
    })

sb.catplot(x="name", y="time", hue="size", data=df, kind="bar")
plt.yscale("log")
#plt.xscale("log")
plt.savefig("simulation-time.png")
