import subprocess
import glob
import os

for s in ["", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
    subprocess.run(
        [
            "cargo",
            "run",
            "-r",
            "--bin=baker",
            "--",
            f"--input=../assets/sphere{s}.glb",
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
            f"--input=../assets/sphere{s}.glb",
            "--output=assets/baker_lod/",
            "--mode=chain",
        ]
    )
    subprocess.run(
        [
            "cargo",
            "run",
            "-r",
            "--bin=baker",
            "--",
            f"--input=../assets/sphere{s}.glb",
            "--output=assets/meshopt_lod/",
            "--mode=chain",
            "--simplifier=meshopt",
        ]
    )
    subprocess.run(
        [
            "cargo",
            "run",
            "-r",
            "--bin=baker",
            "--",
            f"--input=../assets/sphere{s}.glb",
            "--output=assets/meshopt_multires/",
            "--simplifier=meshopt",
        ]
    )

for p in glob.glob("assets/**/sphere*.bin", recursive=True):
    print(p)
    if not os.path.exists(p + ".txt"):
        subprocess.run(
            [
                "cargo",
                "run",
                "-r",
                "--bin=mesh_error",
                "--",
                p,
            ]
        )
