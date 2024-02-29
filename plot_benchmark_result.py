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

forms = {"ExpandingComputeCulledMesh": "-", "IndirectTasks": "-."}
names = {"ExpandingComputeCulledMesh": "DAG Explore", "IndirectTasks": "Task Invoke"}


cols = {"200": "C0", "400": "C1", "600": "C2", "800": "C3", "1000": "C4"}


def def_fig(
    a: bool,
    *s: str,
):
    fig, ax = plt.subplots(1, 1, layout="constrained")

    for mode, result_sets in data.items():
        fig.gca().set_prop_cycle(None)
        for set_name, set in result_sets.items():
            for name, values in set.items():
                if name in s or a:
                    ax.plot(
                        values[:, 0],
                        values[:, 1] * 1000,
                        forms[mode],
                        color=cols[name],
                        label=(name if mode in s else None) if a else names[mode],
                    )

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
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fancybox=True, ncol=3)

    fig.set_size_inches(4, 3)

    fig.savefig(f"../diss/figures/eval/mesh_benchmark{None if a else s}.svg")


def_fig(False, "200")
def_fig(False, "600")
def_fig(False, "1000")
def_fig(True, "ExpandingComputeCulledMesh")

plt.show()
