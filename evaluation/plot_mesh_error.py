import matplotlib.pyplot as plt
import numpy as np


from plot import *

import scienceplots

plt.style.use("science")
plt.rcParams.update({"font.size": 11})

fig, axs = plt.subplots(1, 1, layout="constrained")


def load_data(front: str):
    keys = None
    values = []
    for mesh in [
        "sphere",
        "sphere2",
        "sphere3",
        "sphere4",
        "sphere5",
        "sphere6",
        "sphere7",
        "sphere8",
        "sphere9",
    ]:

        data = np.loadtxt(f"{front}/{mesh}.glb.bin.txt", delimiter=",")
        if keys is not None:
            # print(keys, data[:, 0])
            pass
        else:
            keys = data[1:, 0]

        # axs.plot(data[1:, 0], data[1:, 1], marker="x", label="Test", color="black")

        values.append(data[1:, 1])

    values = np.array(values)
    print("awdawdawd")
    yerr = np.quantile(values, [0.1, 0.9], axis=0)

    plots = np.mean(values, axis=0)

    yerr[0, :] -= plots
    yerr[1, :] -= plots

    return (keys, plots), np.abs(yerr)


multires = load_data("assets")
baker_lod = load_data("assets/baker_lod")
meshopt_lod = load_data("assets/meshopt_lod")
meshopt_multires = load_data("assets/meshopt_multires")

marker = "_"

axs.errorbar(
    *baker_lod[0],
    yerr=baker_lod[1],
    capsize=3,
    marker=marker,
    label="Quadric LOD Chain",
)

axs.errorbar(
    *meshopt_lod[0],
    yerr=meshopt_lod[1],
    marker=marker,
    label="\\texttt{meshopt} LOD Chain",
    capsize=3,
)

axs.errorbar(
    *multires[0], yerr=multires[1], marker=marker, capsize=3, label="Quadric DAG"
)

axs.errorbar(
    *meshopt_multires[0],
    yerr=meshopt_multires[1],
    marker=marker,
    capsize=3,
    label="\\texttt{meshopt} DAG",
)

axs.set_ylim(ymin=0.00000007)
# axs.set_xlim(xmax=multires[:, 0][1])

axs.set_ylabel("Error")
axs.set_xlabel("Triangles")

# axs.set_xscale("log")
axs.set_yscale("log")

config(axs, fig)

plt.savefig(f"../diss/figures/eval/mesh_error.svg")
plt.show()
