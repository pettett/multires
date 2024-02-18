import matplotlib.pyplot as plt
import numpy as np

# mesh = "monk_60k" 
mesh = "sphere" 

multires = np.loadtxt(f"assets/{mesh}.glb.bin.txt", delimiter=",")
baker_lod = np.loadtxt(f"assets/baker_lod/{mesh}.glb.bin.txt", delimiter=",")
meshopt_lod = np.loadtxt(f"assets/meshopt_lod/{mesh}.glb.bin.txt", delimiter=",")

print(multires)

fig, axs = plt.subplots(1, 1, layout='constrained')
axs.plot(multires[:, 0], multires[:, 1], label= "multires")

axs.plot(baker_lod[:, 0], baker_lod[:, 1], label= "baker")

axs.plot(meshopt_lod[:, 0] , meshopt_lod[:, 1], label= "meshopt")

axs.set_ylim(ymin = meshopt_lod[:, 1][1])
axs.set_xlim(xmin = multires[:, 0][-1])

axs.set_xscale("log")
axs.set_yscale("log")

axs.legend()

plt.savefig(f'../diss/figures/eval/{mesh}.svg')
plt.show()