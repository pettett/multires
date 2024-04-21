from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
from plot import *

data: dict[str, dict[str, dict[str, np.ndarray]]] = {}

FRAMETIME = 1
PRIMS = 2
metrics = {FRAMETIME: "Frame time (ms)", PRIMS: "Clipped Primitives"}
scales = {FRAMETIME: 1000, PRIMS: 1}

for d1 in os.listdir("../benchmark/"):
    # print(d1)
    data[d1] = {}

    d2s = os.listdir(f"../benchmark/{d1}/")
    d2 = max(d2s)

    # print(d2)
    data[d1][d2] = {}
    for f in os.listdir(f"../benchmark/{d1}/{d2}"):

        # print(f)

        txt = np.loadtxt(f"../benchmark/{d1}/{d2}/{f}", delimiter=",")
        if txt.shape[1]:  # require PRIMS update
            data[d1][d2][f.split(".")[0]] = txt

dragon = "dragon_high.glb.bin"

expanding_dragon_mesh = f"ExpandingComputeCulledMesh{dragon}"
expanding_torrin_mesh = f"ExpandingComputeCulledMeshtorrin_main.glb.bin"
expanding_dragon_indices = f"ExpandingComputeCulledIndices{dragon}"
local_dragon_mesh = f"LocalSelectMesh{dragon}"
local_dragon_indices = f"LocalSelectIndices{dragon}"

torrin = "ExpandingComputeCulledMeshtorrin_main.glb.bin"
NANITEPRIMS = "Nanite"
NANITETIME = "NaniteTime"

forms = {
    "ExpandingComputeCulledMesh": "-",
    expanding_dragon_mesh: "-",
    expanding_dragon_indices: "-.",
    local_dragon_mesh: "-",
    local_dragon_indices: "-.",
    torrin: "-",
    NANITEPRIMS: "-",
    NANITETIME: "-.",
    "IndirectTasks": "-.",
    "DrawIndirect": "--",
    "DrawLOD": ":",
}
names = {
    "ExpandingComputeCulledMesh": "DAG Explore",
    expanding_dragon_mesh: "Traverse/Mesh",
    expanding_dragon_indices: "Traverse/Primitive",
    local_dragon_mesh: "Local/Mesh",
    local_dragon_indices: "Local/Primitive",
    torrin: "600K Tris",
    NANITEPRIMS: "1000K Tris (Nanite)",
    NANITETIME: "1000K Tris (Nanite)",
    "IndirectTasks": "Task Select",
    "DrawIndirect": "Instanced Full Resolution",
    "DrawLOD": "LOD Chain",
}


cols = {
    expanding_dragon_indices: {
        "500": "C8",
        "1000": "C2",
        "1500": "C4",
        "2000": "C6",
        "2500": "C0",
    },
    expanding_dragon_mesh: {
        "500": "C8",
        "1000": "C2",
        "1500": "C4",
        "2000": "C6",
        "2500": "C0",
    },
    local_dragon_mesh: {
        "500": "C9",
        "1000": "C3",
        "1500": "C5",
        "2000": "C7",
        "2500": "C1",
    },
    local_dragon_indices: {
        "500": "C9",
        "1000": "C3",
        "1500": "C5",
        "2000": "C7",
        "2500": "C1",
    },
    torrin: {
        "500": "C1",
        "1000": "C3",
        "1500": "C5",
        "2000": "C7",
        "2500": "C9",
    },
    NANITEPRIMS: {
        "500": "C1",
        "1000": "C3",
        "1500": "C5",
        "2000": "C7",
        "2500": "C9",
    },
    NANITETIME: {
        "500": "C1",
        "1000": "C3",
        "1500": "C5",
        "2000": "C7",
        "2500": "C9",
    },
    "IndirectTasks": {
        "500": "C1",
        "1000": "C3",
        "1500": "C5",
        "2000": "C7",
        "2500": "C9",
    },
    "DrawLOD": {
        "500": "C1",
        "1000": "C3",
        "1500": "C5",
        "2000": "C7",
        "2500": "C9",
    },
}

maximums = defaultdict(lambda: float("inf"))
markers = defaultdict(lambda: None)

maximums[NANITETIME] = 10.0
markers[NANITEPRIMS] = "x"

# for (nm, ex), (nm2, ts), (nm3, lod) in zip(
#     next(iter(data["ExpandingComputeCulledMesh"].values())).items(),
#     next(iter(data["IndirectTasks"].values())).items(),
#     next(iter(data["DrawLOD"].values())).items(),
# ):

#     assert nm == nm2

#     print(nm)

#     total_ex = np.trapz(ex[:, 0], ex[:, 1])
#     total_ts = np.trapz(ts[:, 0], ts[:, 1])

#     # print(np.mean(ex[:, 1]))
#     # print(np.mean(ts[:, 1]))

#     ex1 = 5 / len(ex[:, 0])
#     # print(np.average(ex[:,1]))
#     ts1 = 5 / len(ts[:, 0])

#     lod1 = 5 / len(lod[:, 0])

#     print(ex1, ts1, lod1)

#     # print("task shader -> expanding", (total_ex - total_ts) / total_ts)
#     print(f"task shader -> expanding {100* (ex1 - ts1) / ts1:2}")


def plot_data(
    ax,
    sample: int,
    a: bool,
    sets: list[str],
    *s: str,
):
    for mode, result_sets in data.items():

        if mode in sets:
            sets.remove(mode)
            for set_name, set in result_sets.items():
                print(set_name)
                for name, values in set.items():
                    print(name)
                    if name in s or a:
                        ax.plot(
                            values[:, 0],
                            np.minimum(
                                values[:, sample] * scales[sample], maximums[mode]
                            ),
                            forms[mode],
                            marker=markers[mode],
                            color=cols[mode][name],
                            label=f"{names[mode]}",
                        )
    print(f"Not seen: {sets}")


def nanite_comparison(
    sample: int,
    a: bool,
    *s: str,
):
    fig, ax = plt.subplots(1, 1, layout="constrained")

    # plot_data(
    #     ax, sample, a, ["ExpandingComputeCulledMesh", "IndirectTasks", "DrawLOD"], *s
    # )
    plot_data(
        ax,
        sample,
        a,
        [expanding_dragon_mesh, NANITETIME, "DrawLOD"],
        *s,
    )

    ax.set_xlabel("Relative Camera Distance")
    ax.set_ylabel(metrics[sample])

    # Put a legend below current axis
    ax.legend(fancybox=True, ncol=2)

    config(None, fig)

    fig.savefig(f"../../diss/figures/eval/mesh_benchmark_nanite.svg")


def internal_comparison(
    sample: int,
    a: bool,
    *s: str,
):
    fig, ax = plt.subplots(1, 1, layout="constrained")

    plot_data(
        ax,
        sample,
        a,
        [
            expanding_dragon_mesh,
            expanding_dragon_indices,
            local_dragon_mesh,
            local_dragon_indices,
        ],
        *s,
    )

    ax.set_xlabel("Relative Camera Distance")
    ax.set_ylabel(metrics[sample])

    ax.set_yscale("log")

    # Put a legend below current axis
    ax.legend(fancybox=True, ncol=2)

    config(None, fig)

    fig.savefig(f"../../diss/figures/eval/mesh_benchmark_internal.svg")


def mesh_nanite_comparison(
    sample: int,
    a: bool,
    *s: str,
):
    fig, ax = plt.subplots(1, 1, layout="constrained")

    # plot_data(
    #     ax, sample, a, ["ExpandingComputeCulledMesh", "IndirectTasks", "DrawLOD"], *s
    # )
    plot_data(
        ax,
        sample,
        a,
        [expanding_dragon_mesh, NANITEPRIMS, "DrawLOD"],
        *s,
    )

    ax.set_xlabel("Relative Camera Distance")
    ax.set_ylabel(metrics[sample])

    # Put a legend below current axis
    ax.legend(fancybox=True, ncol=2)

    config(None, fig)

    fig.savefig(f"../../diss/figures/eval/prims_benchmark_nanite.svg")


def mesh_comparison(
    sample: int,
    a: bool,
    *s: str,
):
    fig, ax = plt.subplots(1, 1, layout="constrained")

    plot_data(
        ax,
        sample,
        a,
        [
            expanding_dragon_mesh,
            expanding_torrin_mesh,
        ],
        *s,
    )

    ax.set_xlabel("Relative Camera Distance")
    ax.set_ylabel(metrics[sample])

    # Put a legend below current axis
    ax.legend(fancybox=True, ncol=2)

    config(None, fig)

    fig.savefig(f"../../diss/figures/eval/prims_benchmark_meshes.svg")


nanite_comparison(FRAMETIME, False, "2500")
internal_comparison(FRAMETIME, False, "2500")

mesh_nanite_comparison(PRIMS, False, "500", "1500", "2500")
mesh_comparison(PRIMS, False, "500", "1500", "2500")


plt.show()
