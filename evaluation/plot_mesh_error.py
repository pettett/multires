import matplotlib.pyplot as plt
import numpy as np

import subprocess
from plot import *

if False:

    subprocess.run(
        [
            "cargo",
            "run",
            "-r",
            "--bin=baker",
            "--",
            "--input=../assets/sphere.glb",
            "--output=assets/",
        ]
    )
    subprocess.run(
        [
            "cargo",
            "run",
            "-r",
            "--bin=baker",
            "--",
            "--input=../assets/sphere.glb",
            "--output=assets/baker_lod/",
            "--mode",
            "baker-lod",
        ]
    )
    subprocess.run(
        [
            "cargo",
            "run",
            "-r",
            "--bin=baker",
            "--",
            "--input=../assets/sphere.glb",
            "--output=assets/meshopt_lod/",
            "--mode",
            "meshopt-lod",
        ]
    )

# mesh = "monk_60k"
mesh = "sphere"

multires = np.loadtxt(f"../assets/{mesh}.glb.bin.txt", delimiter=",")
baker_lod = np.loadtxt(f"../assets/baker_lod/{mesh}.glb.bin.txt", delimiter=",")
meshopt_lod = np.loadtxt(f"../assets/meshopt_lod/{mesh}.glb.bin.txt", delimiter=",")

print(multires)

fig, axs = plt.subplots(1, 1, layout="constrained")

axs.plot(baker_lod[:, 0], baker_lod[:, 1], marker="x", label="Baker LOD Chain")

axs.plot(meshopt_lod[:, 0], meshopt_lod[:, 1], marker="x", label="meshopt LOD Chain")

axs.plot(multires[:, 0], multires[:, 1], marker="x", label="Baker Multiresolution")

axs.set_ylim(ymin=-0.00049, ymax=baker_lod[:, 1][-4])
axs.set_xlim(xmax=multires[:, 0][0])

axs.set_ylabel("Error")
axs.set_xlabel("Triangles")

# axs.set_xscale("log")
# axs.set_yscale("log")

config(axs, fig)

plt.savefig(f"../../diss/figures/eval/mesh_error_{mesh}.svg")
plt.show()
