import matplotlib.pyplot as plt
import numpy as np
import os


data: dict[str, dict[str, dict[str, np.ndarray]]] = {}

FRAMETIME = 1
PRIMS = 2
metrics = {FRAMETIME: "Frame time (ms)", PRIMS: "Clipped Primitives"}
scales = {FRAMETIME: 1000, PRIMS: 1}


for d1 in os.listdir("benchmark/"):
    # print(d1)
    data[d1] = {}

    d2s = os.listdir(f"benchmark/{d1}/")
    d2 = max(d2s)

    # print(d2)
    data[d1][d2] = {}
    for f in os.listdir(f"benchmark/{d1}/{d2}"):

        # print(f)

        txt = np.loadtxt(f"benchmark/{d1}/{d2}/{f}", delimiter=",")
        if txt.shape[1]:  # require PRIMS update
            data[d1][d2][f.split(".")[0]] = txt

dragon = "ExpandingComputeCulledMeshdragon_high.glb.bin"
torrin = "ExpandingComputeCulledMeshtorrin_main.glb.bin"
NANITE = "Nanite"
NANITETIME = "NaniteTime"

forms = {
    "ExpandingComputeCulledMesh": "-",
    dragon: "-",
    torrin: "-",
    NANITE: "-.",
    NANITETIME: "-.",
    "IndirectTasks": "-.",
    "DrawIndirect": "--",
    "DrawLOD": ":",
}
names = {
    "ExpandingComputeCulledMesh": "DAG Explore",
    dragon: "1000K Tris",
    torrin: "600K Tris",
    NANITE: "1000K Tris (Nanite)",
    NANITETIME: "1000K Tris (Nanite)",
    "IndirectTasks": "Task Select",
    "DrawIndirect": "Instanced Full Resolution",
    "DrawLOD": "LOD Chain",
}


cols = {
    "ExpandingComputeCulledMesh": {
        "500": "C0",
        "1000": "C2",
        "1500": "C4",
        "2000": "C6",
        "2500": "C8",
    },
    dragon: {
        "500": "C0",
        "1000": "C2",
        "1500": "C4",
        "2000": "C6",
        "2500": "C8",
    },
    torrin: {
        "500": "C1",
        "1000": "C3",
        "1500": "C5",
        "2000": "C7",
        "2500": "C9",
    },
    NANITE: {
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
                            np.minimum(values[:, sample] * scales[sample], 10.0),
                            forms[mode],
                            color=cols[mode][name],
                            label=f"{names[mode]}-{name}",
                        )
    print(f"Not seen: {sets}")


def def_fig(
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
        [dragon, NANITETIME],
        *s,
    )

    ax.set_xlabel("Relative Camera Distance")
    ax.set_ylabel(metrics[sample])

    # box = ax.get_position()
    # prop = 0.7
    # ax.set_position(
    #     [
    #         box.x0,
    #         box.y0 + box.height * (1 - prop + 0.1),
    #         box.width,
    #         box.height * (prop),
    #     ]
    # )

    # Put a legend below current axis
    ax.legend(fancybox=True, ncol=2)

    fig.set_size_inches(8, 3)

    # fig.savefig(
    #     f"../diss/figures/eval/mesh_benchmark_{metrics[sample]}_{None if a else s}.svg"
    # )
    fig.savefig(f"../diss/figures/eval/mesh_benchmark_{metrics[sample]}.svg")


# def_fig(False, "500")
# def_fig(False, "1000")
# def_fig(False, "2000")
def_fig(FRAMETIME, False, "500", "1500", "2500")

# # fig, [ax1, ax] = plt.subplots(2, sharex=True)

# # plot_data(ax1, False, ["ExpandingComputeCulledMesh", "IndirectTasks"], "500")


# # def first_data(result_sets):
# #     for set_name, set in result_sets.items():
# #         for name, values in set.items():
# #             return values


# # compute = first_data(data["ExpandingComputeCulledMesh"])
# # task = first_data(data["IndirectTasks"])

# # q = np.arange(0, 1.02, 0.01)
# # compute_digi = np.digitize(compute[:, 0], q)
# # task_digi = np.digitize(task[:, 0], q)

# # compute_ft = np.zeros_like(q, dtype=float)
# # task_ft = np.zeros_like(q, dtype=float)

# # for i in range(len(task_digi)):
# #     task_ft[task_digi[i]] = max(task_ft[task_digi[i]], task[i, 1])

# # for i in range(len(compute_digi)):
# #     compute_ft[compute_digi[i]] = max(compute_ft[compute_digi[i]], compute[i, 1])

# # diff = compute_ft - task_ft

# # ax.plot(q, diff, label="Difference")


# # ax.set_xlabel("Relative Camera Distance")
# # ax.set_ylabel("Frame time (ms)")


# # ax1.legend()
# # ax.legend()

# # fig.set_size_inches(4, 3)

# # fig.savefig(f"../diss/figures/eval/compare_methods.svg")


plt.show()
