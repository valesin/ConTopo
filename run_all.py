import json
import itertools
import subprocess
import sys

CONFIG_PATH = "configs/experiments_cross_entropy_torus.json"

def main():
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    python_exec = cfg.get("defaults", {}).get("python", sys.executable)
    experiments = cfg["experiments"]

    commands = []

    for exp in experiments:
        script = exp["script"]
        fixed_args = exp.get("fixed_args", [])
        fixed_flags = exp.get("flags", {})
        grid = exp.get("grid", {})

        grid_keys = list(grid.keys())
        grid_values = [grid[k] for k in grid_keys] if grid_keys else [()]

        for combo in (itertools.product(*grid_values) if grid_keys else [()]):
            cmd = [python_exec, script, *fixed_args]

            for k, v in fixed_flags.items():
                cmd += [f"--{k}", str(v)]

            if grid_keys:
                for k, v in zip(grid_keys, combo):
                    cmd += [f"--{k}", str(v)]

            commands.append(cmd)

    print(f"Prepared {len(commands)} runs. Starting...\n")

    for i, cmd in enumerate(commands, 1):
        pretty = " ".join(cmd)
        print(f"[{i:03d}/{len(commands)}] {pretty}")
        result = subprocess.run(cmd)
        print(f" -> return code: {result.returncode}\n")

    print("All done.")

if __name__ == "__main__":
    main()