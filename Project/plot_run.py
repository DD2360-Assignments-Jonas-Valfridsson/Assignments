import matplotlib.pyplot as plt
import subprocess
import seaborn as sb
import numpy as np
import pandas as pd

original, mini_op, with_floats, mini_op_floats = [], [], [], []
for _ in np.arange(3):

    original_output = subprocess.check_output(
        f"./cuda_mandel 0", shell=True).decode().splitlines()
    floats_output = subprocess.check_output(f"./cuda_mandel_floats 0",
                                            shell=True).decode().splitlines()
    mini_op_output = subprocess.check_output(f"./cuda_mandel_minimize_op 0",
                                             shell=True).decode().splitlines()
    mini_op_floats_output = subprocess.check_output(
        f"./cuda_mandel_all_optimization 0", shell=True).decode().splitlines()

    print(original_output, floats_output, mini_op_output, mini_op_floats_output)

    original.append(int(original_output[0].split()[1]))
    with_floats.append(int(floats_output[0].split()[1]))
    mini_op.append(int(mini_op_output[0].split()[1]))
    mini_op_floats.append(int(mini_op_floats_output[0].split()[1]))

data = pd.DataFrame({
    "optimization": ["None"] * len(original) + ["registers"] * len(mini_op) + ["floats"] * len(with_floats) +
     ["both"] * len(mini_op_floats),
    "time in ms":
    original + mini_op + with_floats + mini_op_floats
})

plt.figure(figsize=(10, 6))
sb.barplot(x="optimization", y="time in ms", data=data)
plt.savefig("run.png", bbox_inches="tight")
plt.show()
