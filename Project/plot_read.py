import matplotlib.pyplot as plt
import subprocess
import seaborn as sb
import numpy as np
import pandas as pd

pagable, pinned = [], []
for _ in np.arange(5):

    pagable_output = subprocess.check_output(f"./cuda_mandel 0", shell=True).decode().splitlines()
    pinned_output = subprocess.check_output(f"./cuda_mandel_pinned 0", shell=True).decode().splitlines()

    print("Pageable", pagable_output)
    print("Pinned", pinned_output)


    pagable.append(int(pagable_output[1].split()[1]))
    pinned.append(int(pinned_output[1].split()[1]))


data = pd.DataFrame({
    "memory type": ["Pageable"] * len(pagable) + ["Pinned"] * len(pinned),
    "time in ms": pagable + pinned
    })



plt.figure(figsize=(10, 6))
sb.barplot(x="memory type", y="time in ms", data=data)
plt.savefig("read.png", bbox_inches="tight")
plt.show()


