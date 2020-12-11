import matplotlib.pyplot as plt
import subprocess
import seaborn as sb
import numpy as np
import pandas as pd


def get_gpu_time(output):
    return int(output[0].split()[1]) + int(output[1].split()[1])


def get_cpu_time(output):
    return int(output[0].split()[1])


cpu, ori, pinned, register, precision, all = [], [], [], [], [], []
for _ in np.arange(3):

    ori.append(
        get_gpu_time(
            subprocess.check_output(f"./cuda_mandel 0",
                                    shell=True).decode().splitlines()))

    pinned.append(
        get_gpu_time(
            subprocess.check_output(f"./cuda_mandel_pinned 0",
                                    shell=True).decode().splitlines()))

    precision.append(
        get_gpu_time(
            subprocess.check_output(f"./cuda_mandel_floats 0",
                                    shell=True).decode().splitlines()))

    register.append(
        get_gpu_time(
            subprocess.check_output(f"./cuda_mandel_minimize_op 0",
                                    shell=True).decode().splitlines()))
    all.append(
        get_gpu_time(
            subprocess.check_output(f"./cuda_mandel_all_optimization 0",
                                    shell=True).decode().splitlines()))

    cpu.append(
        get_cpu_time(
            subprocess.check_output(f"./normal_mandel 0",
                                    shell=True).decode().splitlines()))

data = pd.DataFrame({
    "code":
    ["cpu"] * len(cpu) + ["gpu"] * len(ori) + ["gpu-pinned"] * len(pinned) + ["gpu-reg"] * len(register) +
    ["gpu-f32"] * len(precision) + ["gpu-all"] * len(all),
    "time in ms":
    cpu + ori + pinned + register + precision + all
})

plt.figure(figsize=(10, 6))
sb.barplot(x="code", y="time in ms", data=data)
plt.savefig("cpu_vs_gpu.png", bbox_inches="tight")
plt.show()
