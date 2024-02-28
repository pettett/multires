import matplotlib.pyplot as plt
import numpy as np
import os


data: dict[str, dict[str, dict[str, np.ndarray]]] = {}

for d1 in os.listdir("benchmark/"):
    print(d1)
    data[d1] = {}

    for d2 in os.listdir(f"benchmark/{d1}/"):
        print(d2)
        data[d1][d2] = {}
        for f in os.listdir(f"benchmark/{d1}/{d2}"):

            print(f)
            data[d1][d2][f] = np.loadtxt(f"benchmark/{d1}/{d2}/{f}", delimiter=",")

forms = {"ExpandingComputeCulledMesh": "-", "IndirectTasks": "-."}

fig, axs = plt.subplots(1, 1, layout="constrained")

for mode, result_sets in data.items():
    for set_name, set in result_sets.items():
        for name, values in set.items():

            axs.plot(values[:, 0], values[:, 1], forms[mode], label=name)

# axs.set_ylim(ymin=meshopt_lod[:, 1][1])
# axs.set_xlim(xmin=multires[:, 0][-1])

# axs.set_xscale("log")
# axs.set_yscale("log")

axs.legend()

plt.savefig(f"../diss/figures/eval/mesh_benchmark.svg")
plt.show()
