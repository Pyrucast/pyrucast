"""Run the pyrucast script of every validation case and store its values.

Each `run.py` is run in its own process, from its own folder, so that the
figures it writes land next to it. Its last printed line is the dictionary
returned by its `main`; it is stored in `results.json`.

Running::

    python validation/run_pyrucast.py            # every case with a run.py
    python validation/run_pyrucast.py elas8 ...  # only these cases
"""

import ast
import json
import pathlib
import subprocess
import sys

import pyrucast

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    wanted = set(sys.argv[1:])
    scripts = sorted(ROOT.glob("*/*/run.py"))
    failed = []
    for script in scripts:
        folder = script.parent
        if wanted and folder.name not in wanted:
            continue
        proc = subprocess.run(
            [sys.executable, "run.py"],
            cwd=folder,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            failed.append(folder.name)
            print(f"{folder.name}: FAILED\n{proc.stderr[-2000:]}")
            continue
        values = ast.literal_eval(proc.stdout.strip().splitlines()[-1])
        record = {
            "pyrucast_version": pyrucast.__version__,
            "values": {key: float(value) for key, value in values.items()},
        }
        text = json.dumps(record, indent=2) + "\n"
        (folder / "results.json").write_text(text, encoding="utf-8")
        print(f"{folder.name}: {record['values']}")
    if failed:
        sys.exit(f"failed: {' '.join(failed)}")


if __name__ == "__main__":
    main()
