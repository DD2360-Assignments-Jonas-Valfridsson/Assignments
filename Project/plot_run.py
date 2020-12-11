import matplotlib.pyplot as plt
import subprocess
import seaborn as sb
import numpy as np
import pandas as pd


def get_gpu_time(output):
    return int(output[0].split()[1])


ori, pinned, register, precision, all = [], [], [], [], []
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
            subprocess.check_output(f"./cuda_mandel_register 0",
                                    shell=True).decode().splitlines()))
    all.append(
        get_gpu_time(
            subprocess.check_output(f"./cuda_mandel_all 0",
                                    shell=True).decode().splitlines()))

data = pd.DataFrame({
    "optimization": ["gpu"] * len(ori) + ["gpu-reg"] * len(register) +
    ["gpu-f32"] * len(precision) + ["gpu-all"] * len(all),
    "time in ms":
    ori + register + precision + all
})

plt.figure(figsize=(10, 6))
sb.barplot(x="optimization", y="time in ms", data=data)
plt.savefig("run.png", bbox_inches="tight")
plt.show()
