import matplotlib.pyplot as plt
import numpy as np
import os


data: dict[str, dict[str, dict[str, np.ndarray]]] = {}

for d1 in os.listdir("benchmark/"):
    print(d1)
    data[d1] = {}

    d2s = os.listdir(f"benchmark/{d1}/")
    d2 = max(d2s)

    print(d2)
    data[d1][d2] = {}
    for f in os.listdir(f"benchmark/{d1}/{d2}"):

        print(f)
        data[d1][d2][f.split(".")[0]] = np.loadtxt(
            f"benchmark/{d1}/{d2}/{f}", delimiter=","
        )

forms = {
    "ExpandingComputeCulledMesh": "-",
    "IndirectTasks": "-.",
    "DrawIndirect": "--",
    "DrawLOD": "-",
}
names = {
    "ExpandingComputeCulledMesh": "DAG Explore",
    "IndirectTasks": "Task Select",
    "DrawIndirect": "Instanced Full Resolution",
    "DrawLOD": "CPU Controlled LOD Chain",
}
alphas = {
    "ExpandingComputeCulledMesh": 1.0,
    "IndirectTasks": 1.0,
    "DrawIndirect": 1.0,
    "DrawLOD": 1.0,
}

cols = {
    "ExpandingComputeCulledMesh": {
        "500": "C0",
        "1000": "C2",
        "1500": "C4",
        "2000": "C6",
        "2500": "C8",
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

for (nm, ex), (nm2, ts), (nm3, lod) in zip(
    next(iter(data["ExpandingComputeCulledMesh"].values())).items(),
    next(iter(data["IndirectTasks"].values())).items(),
    next(iter(data["DrawLOD"].values())).items(),
):

    assert nm == nm2

    print(nm)

    total_ex = np.trapz(ex[:, 0], ex[:, 1])
    total_ts = np.trapz(ts[:, 0], ts[:, 1])

    # print(np.mean(ex[:, 1]))
    # print(np.mean(ts[:, 1]))

    ex1 = 5 / len(ex[:, 0])
    # print(np.average(ex[:,1]))
    ts1 = 5 / len(ts[:, 0])

    lod1 = 5 / len(lod[:, 0])

    print(ex1, ts1, lod1)

    # print("task shader -> expanding", (total_ex - total_ts) / total_ts)
    print(f"task shader -> expanding {100* (ex1 - ts1) / ts1:2}")


def plot_data(
    ax,
    a: bool,
    sets: set,
    *s: str,
):
    for mode, result_sets in data.items():

        if mode in sets:
            for set_name, set in result_sets.items():
                for name, values in set.items():
                    if name in s or a:
                        ax.plot(
                            values[::5, 0],
                            values[::5, 1] * 1000,
                            forms[mode],
                            color=cols[mode][name],
                            alpha=alphas[mode],
                            label=f"{names[mode]}-{name}",
                        )


def def_fig(
    a: bool,
    *s: str,
):
    fig, ax = plt.subplots(1, 1, layout="constrained")

    plot_data(ax, a, ["ExpandingComputeCulledMesh", "IndirectTasks", "DrawLOD"], *s)

    ax.set_xlabel("Relative Camera Distance")
    ax.set_ylabel("Frame time (ms)")

    box = ax.get_position()
    prop = 0.7
    ax.set_position(
        [
            box.x0,
            box.y0 + box.height * (1 - prop + 0.1),
            box.width,
            box.height * (prop),
        ]
    )

    # Put a legend below current axis
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fancybox=True, ncol=2)

    fig.set_size_inches(4, 3)

    fig.savefig(f"../diss/figures/eval/mesh_benchmark{None if a else s}.svg")


# def_fig(False, "500")
# def_fig(False, "1000")
# def_fig(False, "2000")
def_fig(False, "500", "1500", "2500")

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
