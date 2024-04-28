from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import os
from plot import *

data: dict[str, dict[str, dict[str, np.ndarray]]] = {}

FRAMETIME = "false"
PRIMS = "true"
metrics = {FRAMETIME: "Frame time (ms)", PRIMS: "Clipped Primitives"}
samples = {FRAMETIME: 1, PRIMS: 2}


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

    def to_string(self):
        return f"{self.selector}{self.geometry}{self.mesh}{self.bench}{self.error}"

    def sample(self):
        match self.bench:
            case x if x == PRIMS:
                return 2
            case _:
                return 1

    def form(self):
        return "-"

    def marker(self):
        if self.geometry == NANITE and self.bench == PRIMS:
            return "x"
        return ""


torrin = "ExpandingComputeCulledMeshtorrin_main.glb.bin"
NANITEPRIMS = "Nanite"
NANITETIME = "NaniteTime"


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
    a: bool,
    benchmarks: list[Benchmark],
    *s: str,
):

    for data_name, result_sets in data.items():

        for benchmark in benchmarks:
            if benchmark.to_string() == data_name:

                benchmarks.remove(benchmark)
                for set_name, set in result_sets.items():
                    print(set_name)
                    for name, values in set.items():
                        # print(name)
                        if name in s or a:
                            ax.plot(
                                values[:, 0],
                                np.minimum(
                                    values[:, benchmark.sample()],
                                    maximums[data_name],
                                ),
                                benchmark.form(),
                                marker=benchmark.marker(),
                                # color=cols[benchmark.selector][name],
                                label=benchmark.title,
                            )
                break

    print(f"Not seen: {benchmarks}")


def nanite_comparison(
    a: bool,
    *s: str,
):
    fig, ax = plt.subplots(1, 1, layout="constrained")

    # plot_data(
    #     ax, sample, a, ["ExpandingComputeCulledMesh", "IndirectTasks", "DrawLOD"], *s
    # )
    plot_data(
        ax,
        a,
        [
            Benchmark(
                LOCAL, MESH, DRAGON, FRAMETIME, ERR01, "Ours (Mesh Shading), τ=0.1"
            ),
            Benchmark(
                LOCAL, MESH, DRAGON, FRAMETIME, ERR02, "Ours (Mesh Shading), τ=0.2"
            ),
            Benchmark(NANITE, NONE, NONE, FRAMETIME, NONE, "Nanite"),
            Benchmark(DRAWLOD, NONE, NONE, NONE, NONE, "LOD Chain"),
        ],
        *s,
    )

    ax.set_xlabel("Relative Camera Distance")
    ax.set_ylabel(metrics[FRAMETIME])

    # Put a legend below current axis
    ax.legend(fancybox=True, ncol=2)

    config(None, fig)

    fig.savefig(f"../diss/figures/eval/mesh_benchmark_nanite.svg")


def internal_comparison(
    a: bool,
    *s: str,
):
    fig, ax = plt.subplots(1, 1, layout="constrained")

    plot_data(
        ax,
        a,
        [
            Benchmark(
                LOCAL, MESH, DRAGON, FRAMETIME, ERR01, "Local Select + Mesh Shading"
            ),
            Benchmark(
                TRAVERSE,
                MESH,
                DRAGON,
                FRAMETIME,
                ERR01,
                "Traverse Select + Mesh Shading",
            ),
            Benchmark(
                LOCAL,
                INDICES,
                DRAGON,
                FRAMETIME,
                ERR01,
                "Local Select + Primitive Shading",
            ),
            Benchmark(
                TRAVERSE,
                INDICES,
                DRAGON,
                FRAMETIME,
                ERR01,
                "Traverse Select + Primitive Shading",
            ),
        ],
        *s,
    )

    ax.set_xlabel("Relative Camera Distance")
    ax.set_ylabel(metrics[FRAMETIME])

    ax.set_yscale("log")

    # Put a legend below current axis
    ax.legend(fancybox=True, ncol=2)

    config(None, fig)
    fig.savefig(f"../diss/figures/eval/mesh_benchmark_internal.svg")


def mesh_nanite_comparison(
    a: bool,
    *s: str,
):
    fig, ax = plt.subplots(1, 1, layout="constrained")

    # plot_data(
    #     ax, sample, a, ["ExpandingComputeCulledMesh", "IndirectTasks", "DrawLOD"], *s
    # )
    plot_data(
        ax,
        a,
        [
            Benchmark(LOCAL, MESH, DRAGON, PRIMS, ERR01, "Ours, τ=0.1"),
            Benchmark(LOCAL, MESH, DRAGON, PRIMS, ERR02, "Ours, τ=0.2"),
            Benchmark(NANITE, NONE, NONE, PRIMS, NONE, "Nanite"),
            # Benchmark(DRAWLOD, NONE, NONE, NONE, NONE),
        ],
        *s,
    )

    ax.set_xlabel("Relative Camera Distance")
    ax.set_ylabel(metrics[PRIMS])

    # Put a legend below current axis
    ax.legend(fancybox=True, ncol=2)

    config(None, fig)

    fig.savefig(f"../diss/figures/eval/prims_benchmark_nanite.svg")


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


nanite_comparison(False, "2500")
internal_comparison(False, "2500")

mesh_nanite_comparison(False, "2500")
mesh_comparison(False, "2500")


plt.show()
