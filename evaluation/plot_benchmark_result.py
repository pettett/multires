from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import os
from plot import *

from scipy.stats import linregress, t
from scipy import stats

import scienceplots

plt.style.use("science")
plt.rcParams.update({"font.size": 11})


FRAMETIME = "false"
PRIMS = "true"
metrics = {FRAMETIME: "Frame time (ms)", PRIMS: "Visible Primitives"}
samples = {FRAMETIME: 1, PRIMS: 2}


def load_data(path: str) -> dict[str, dict[str, dict[str, np.ndarray]]]:

    data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for d1 in os.listdir(f"{path}/"):
        # print(d1)
        data[d1] = {}

        d2s = os.listdir(f"{path}/{d1}/")
        d2 = max(d2s)

        # print(d2)
        data[d1][d2] = {}
        for f in os.listdir(f"{path}/{d1}/{d2}"):

            # print(f)
            txt = np.loadtxt(f"{path}/{d1}/{d2}/{f}", delimiter=",")
            if txt.shape[1]:  # require PRIMS update
                data[d1][d2][f.split(".")[0]] = txt
    return data


data = load_data("benchmark")

DRAGON = "dragon_high.glb.bin"
TORRIN = "torrin_main.glb.bin"

NANITE = "Nanite"
MESH = "Mesh"
INDICES = "Indices"
NONE = ""

LOCAL = "LocalSelect"
TRAVERSE = "ExpandingComputeCulled"
DRAWLOD = "DrawLOD"
DRAWINDIRECT = "DrawIndirect"

ERR01 = "0.1"
ERR02 = "0.2"


@dataclass(frozen=True)
class Benchmark:
    selector: str
    geometry: str
    mesh: str
    bench: str
    error: str
    title: str
    col: str = None
    dashing: str = None
    scale: float = 1.0
    marker: str = None
    sample_o: int = None
    yerr: list = None

    def to_string(self):
        return f"{self.selector}{self.geometry}{self.mesh}{self.bench}{self.error}"

    def sample(self):
        if self.sample_o != None:
            return self.sample_o

        match self.bench:
            case x if x == PRIMS:
                return 2
            case _:
                return 1

    def form(self):
        return self.dashing if self.dashing else "-"


torrin = "ExpandingComputeCulledMeshtorrin_main.glb.bin"
NANITEPRIMS = "Nanite"
NANITETIME = "NaniteTime"
NANITEWORSTTIME = "NaniteWorstTime"


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


def find_nearest(array: np.ndarray, value: float) -> int:
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def get_benchmark_data(benchmark: Benchmark, data):
    for data_name, result_sets in data.items():

        if benchmark.to_string() == data_name:
            return next(iter(result_sets.values()))


def plot_data(
    ax,
    benchmarks: list[Benchmark],
    *s: str,
):

    for benchmark in benchmarks:
        set = get_benchmark_data(benchmark, data)
        runs = []
        for name, values in set.items():
            # print(name)
            if name in s or name in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
                runs.append(values)

        runs = np.array(runs)

        values = np.mean(runs, axis=0)

        print(benchmark.sample())

        minv = np.min(runs[:, :, benchmark.sample()], axis=0) * benchmark.scale
        maxv = np.max(runs[:, :, benchmark.sample()], axis=0) * benchmark.scale

        ax.plot(
            values[:, 0],
            minv,
            benchmark.form(),
            marker=benchmark.marker,
            color=benchmark.col,
            label=benchmark.title,
        )
        if len(runs) > 1:
            ax.fill_between(values[:, 0], minv, maxv, alpha=0.1, color=benchmark.col)
        elif benchmark.yerr != None:
            ax.fill_between(
                values[:, 0],
                minv - benchmark.yerr[0],
                minv + benchmark.yerr[1],
                alpha=0.1,
                color=benchmark.col,
            )


def nanite_comparison(
    *s: str,
):
    fig, ax = plt.subplots(1, 1, layout="constrained")

    # plot_data(
    #     ax, sample, a, ["ExpandingComputeCulledMesh", "IndirectTasks", "DrawLOD"], *s
    # )
    plot_data(
        ax,
        [
            Benchmark(
                LOCAL,
                MESH,
                DRAGON,
                FRAMETIME,
                ERR01,
                "Mine (mesh shading), $\\tau=0.1$",
                col="C0",
            ),
            Benchmark(
                LOCAL,
                MESH,
                DRAGON,
                FRAMETIME,
                ERR02,
                "Mine (mesh shading), $\\tau=0.2$",
                col="C1",
            ),
            Benchmark(
                NANITEWORSTTIME,
                NONE,
                NONE,
                NONE,
                NONE,
                "Unreal Engine",
                yerr=[0.04203714799280989, 0.21796285200719012],
                col="C3",
            ),
            Benchmark(
                DRAWLOD,
                NONE,
                DRAGON,
                FRAMETIME,
                "0.05",
                "LOD Chain",
                sample_o=3,
                scale=1000,
                col="C4",
            ),
        ],
        *s,
    )

    ax.set_ylim(bottom=0, top=7)

    ax.set_xlabel("Relative Camera Distance")
    ax.set_ylabel("Render Time (ms)")

    # Put a legend below current axis
    ax.legend(fancybox=True, ncol=2)

    config(None, fig)

    fig.savefig(f"../diss/figures/eval/mesh_benchmark_nanite.svg")


def internal_comparison(
    *s: str,
):
    fig, ax = plt.subplots(1, 1, layout="constrained")

    ax.fill_between(
        [0, 1], [0, 0], [1000 / 60, 1000 / 60], label="$>60$FPS", color="g", alpha=0.05
    )

    plot_data(
        ax,
        [
            Benchmark(
                LOCAL,
                MESH,
                DRAGON,
                FRAMETIME,
                ERR01,
                "Adaptive Select + Mesh Shading",
                col="C3",
            ),
            Benchmark(
                TRAVERSE,
                MESH,
                DRAGON,
                FRAMETIME,
                ERR01,
                "DAG Traverse + Mesh Shading",
                col="C2",
            ),
            Benchmark(
                LOCAL,
                INDICES,
                DRAGON,
                FRAMETIME,
                ERR01,
                "Adaptive Select + Primitive Shading",
                col="C3",
                dashing="--",
            ),
            Benchmark(
                TRAVERSE,
                INDICES,
                DRAGON,
                FRAMETIME,
                ERR01,
                "DAG Traverse + Primitive Shading",
                col="C2",
                dashing="--",
            ),
        ],
        *s,
    )

    ax.set_xlabel("Relative Camera Distance")
    ax.set_ylabel("GPU Time (ms)")

    ax.set_yscale("log")
    ax.set_ylim(top=500)

    ax.set_xlim(left=0, right=1)

    # Put a legend below current axis
    ax.legend(fancybox=True, ncol=2, loc="upper left")

    config(None, fig)
    fig.savefig(f"../diss/figures/eval/mesh_benchmark_internal.svg")


def mesh_nanite_comparison(
    *s: str,
):
    fig, ax = plt.subplots(1, 1, layout="constrained")

    ax.hlines(1920 * 1080, 0, 1, color="red", label="$1080\\times 1920$ Pixels")

    # plot_data(
    #     ax, sample, a, ["ExpandingComputeCulledMesh", "IndirectTasks", "DrawLOD"], *s
    # )
    plot_data(
        ax,
        [
            Benchmark(
                LOCAL,
                MESH,
                DRAGON,
                PRIMS,
                ERR01,
                "Ours, $\\tau=0.1$",
                col="C0",
            ),
            Benchmark(
                LOCAL,
                MESH,
                DRAGON,
                PRIMS,
                ERR02,
                "Ours, $\\tau=0.2$",
                col="C0",
                dashing="-.",
            ),
            # Benchmark(
            #     LOCAL,
            #     MESH,
            #     TORRIN,
            #     PRIMS,
            #     ERR01,
            #     "Ours, 600K Tris/Mesh, $\\tau=0.1$",
            #     col="C1",
            # ),
            Benchmark(NANITE, NONE, NONE, PRIMS, NONE, "Nanite", marker="x", col="C2"),
            # Benchmark(DRAWLOD, NONE, NONE, NONE, NONE),
        ],
        *s,
    )
    ax.set_ylim(bottom=0, top=1920 * 1080 * 10)
    ax.set_xlabel("Relative Camera Distance")
    ax.set_ylabel(metrics[PRIMS])

    # Put a legend below current axis
    ax.legend(fancybox=True, ncol=2)

    config(None, fig)

    fig.set_size_inches(PLOT_SIZE[0], PLOT_SIZE[1])

    fig.savefig(f"../diss/figures/eval/prims_benchmark_nanite.svg")


def mesh_lines_comparison(
    a: bool,
    *s: str,
):
    fig, [ax, ax2] = plt.subplots(1, 2, layout="constrained", sharey=True)

    b1 = Benchmark(
        LOCAL,
        MESH,
        DRAGON,
        PRIMS,
        ERR01,
        "$T=1000K, \\tau=0.1$",
        col="C0",
    )

    b2 = Benchmark(
        LOCAL,
        MESH,
        DRAGON,
        PRIMS,
        ERR02,
        "$T=1000K, \\tau=0.2$",
        col="C0",
    )

    b3 = Benchmark(
        LOCAL,
        MESH,
        TORRIN,
        PRIMS,
        ERR01,
        "$T=600K, \\tau=0.1$",
        col="C0",
    )

    # Put a legend below current axis
    ax.legend(fancybox=True, ncol=2)

    d1 = get_benchmark_data(b1, data)["2500"]
    d2 = get_benchmark_data(b2, data)["2500"]
    d3 = get_benchmark_data(b3, data)["2500"]

    data2 = load_data("benchmark_rand")

    b21 = Benchmark(
        TRAVERSE,
        MESH,
        DRAGON,
        PRIMS,
        ERR01,
        "",
    )

    b22 = Benchmark(
        TRAVERSE,
        MESH,
        TORRIN,
        PRIMS,
        ERR01,
        "",
    )

    d21 = get_benchmark_data(b21, data2)["2500"]
    d22 = get_benchmark_data(b22, data2)["2500"]
    # d22 = get_benchmark_data(b2, data2)["2500"]
    # d23 = get_benchmark_data(b3, data2)["2500"]

    print(d21)

    for i in range(len(d2)):
        nearest = find_nearest(d1[:, 0], d2[i, 0])
        d2[i, 0] = d1[nearest, 2]

    for i in range(len(d3)):
        nearest = find_nearest(d1[:, 0], d3[i, 0])
        d3[i, 0] = d1[nearest, 2]

    x1 = d21[:, 3]
    y = d21[:, 2]
    x2 = d22[:, 2]
    coef2 = np.polyfit(x1, y, 1)
    coef3 = np.polyfit(x2, y, 1)

    print(linregress(x1, y), coef2)
    print(linregress(x2, y), coef3)

    # ax.plot(d2[::10, 2], d2[::10, 0], color="C0", label="Benchmark Path")
    ax.scatter(x1, y, marker=".", color="C0", label="Random Viewpoints", alpha=0.2)
    # ax2.plot(d3[::10, 2], d3[::10, 0], color="C2", label="Benchmark Path")
    ax2.scatter(x2, y, marker=".", color="C2", label="Random Viewpoints", alpha=0.2)

    config(None, fig)

    fig.set_size_inches(PLOT_SIZE[0], PLOT_SIZE[1] - 0.4)

    # poly1d_fn is now a function which takes in x and returns an estimate for y
    poly1d_fn2 = np.poly1d(coef2)
    poly1d_fn3 = np.poly1d(coef3)

    # https://tomholderness.wordpress.com/2013/01/10/confidence_intervals/
    fit = poly1d_fn2(x1)
    c_y = [np.min(fit), np.max(fit)]
    c_x = [np.min(x1), np.max(x1)]

    ax.plot(
        c_x, c_y, "--k", label="Regression Line"
    )  #'--k'=black dashed line, 'yo' = yellow circle marker

    fit = poly1d_fn3(x2)
    c_y = [np.min(fit), np.max(fit)]
    c_x = [np.min(x2), np.max(x2)]

    ax2.plot(
        c_x, c_y, "--k", label="Regression Line"
    )  #'--k'=black dashed line, 'yo' = yellow circle marker

    ax.set_ylabel(b1.title)
    ax.set_xlabel(b2.title)
    ax2.set_xlabel(b3.title)

    ax.ticklabel_format(style="sci", scilimits=(6, 6), axis="both")
    ax2.ticklabel_format(style="sci", scilimits=(6, 6), axis="both")

    ax.legend(loc="upper left")
    ax2.legend(loc="upper left")

    ax.set_title("Same mesh, different error")
    ax2.set_title("Different mesh, same error")

    fig.savefig(f"../diss/figures/eval/prims_benchmark_lines.svg")


def mesh_comparison(
    a: bool,
    *s: str,
):
    fig, ax = plt.subplots(1, 1, layout="constrained")

    plot_data(
        ax,
        a,
        [
            Benchmark(LOCAL, MESH, DRAGON, PRIMS, ERR01, "1000K Tris/Mesh"),
            Benchmark(LOCAL, MESH, TORRIN, PRIMS, ERR01, "600K Tris/Mesh"),
        ],
        *s,
    )

    ax.set_xlabel("Relative Camera Distance")
    ax.set_ylabel(metrics[PRIMS])

    # Put a legend below current axis
    ax.legend(fancybox=True, ncol=2)

    config(None, fig)

    fig.savefig(f"../diss/figures/eval/prims_benchmark_meshes.svg")


def stress_test():
    fig, ax = plt.subplots(1, 1, layout="constrained")

    # Manually gathered data over different sizes

    bench = Benchmark("scene_LocalSelect", MESH, DRAGON, PRIMS, ERR01, "600K Tris/Mesh")

    bench_data = get_benchmark_data(bench, data)

    datapoints = np.zeros(len(bench_data))
    datapoints_prims = np.zeros_like(datapoints)
    datapoints_times = np.zeros_like(datapoints)
    datapoints_errs = np.zeros((2, len(bench_data)))

    for instance_count, d in bench_data.items():
        instance_count = int(instance_count)
        i = instance_count // 2500 - 1

        datapoints[i] = instance_count * 1000000

        datapoints_prims[i] = d[0, 2]
        datapoints_times[i] = np.average(d[:, 1])
        datapoints_errs[:, i] = np.abs(
            np.quantile(d[:, 1], [0.1, 0.9]) - datapoints_times[i]
        )

    ax.set_title("Scene Complexity")

    ax.ticklabel_format(style="sci", scilimits=(9, 9), axis="x")
    ax.errorbar(
        datapoints[1:],
        datapoints_times[1:],
        datapoints_errs[:, 1:],
        capsize=3,
        # marker=".",
        label="GPU Time",
    )

    ax2 = ax.twinx()

    ax.set_xlabel("Scene Triangles")
    ax.set_ylabel("GPU Time (ms)")

    ax2.set_ylabel(metrics[PRIMS])

    ax2.ticklabel_format(style="sci", scilimits=(6, 6), axis="y")
    ax2.plot([], [], label="GPU Time")
    ax2.plot(datapoints[1:], datapoints_prims[1:], label="Triangles Rasterised")

    # Put a legend below current axis
    ax2.legend(fancybox=True, ncol=2, loc="upper left")

    config(None, fig)
    ax.set_ylim(bottom=0.0)
    ax2.set_ylim(bottom=0.0)

    fig.savefig(f"../diss/figures/eval/stress_test.svg")


# nanite_comparison("2500")
internal_comparison("2500")

# stress_test()

# mesh_nanite_comparison("2500")
# mesh_lines_comparison("2500")
# mesh_comparison("2500")


plt.show()
