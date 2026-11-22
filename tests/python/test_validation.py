"""The validation cases of `validation/`: each computed case meets its tolerance.

Every folder of `validation/` holding a `run.py` reproduces a case of Cast3M's
"Annotated Testing Files". This test runs each script and compares its values
with the case's references and tolerances, with the very check that writes the
verdicts of `validation/README.md` (`validation/make_document.py`).

A case known to miss its tolerance is listed in `EXPECTED_FAILURES`, with the
reason: the improvement point of pyrucast it waits for. It must keep failing —
once it passes, it leaves the list.
"""

import importlib.util
import pathlib
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2] / "validation"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DOCUMENT = _load(ROOT / "make_document.py", "validation_make_document")
CASES = sorted(p.parent for p in ROOT.glob("*/*/run.py"))

EXPECTED_FAILURES = {
    "elas2": "`convert` cuts the quadrangles along one diagonal, the `.dgibi` "
    "file alternates it; the criterion (6e-4) is tighter than that",
    "elas3": "the layers are even (no graded extrusion) and the diagonals alike; "
    "the criterion is 4e-4",
    "elas8": "`transfinite` puts the QUA8 mid-nodes of the ellipses on the chords",
    "plas5": "the two-surface Drucker-Prager stays on its initial surface",
    "plas7": "a stress read at a node in one cell shows the locked QUA4's "
    "oscillation (no nodal averaging, no B-bar)",
    "plas9": "the two-surface Drucker-Prager does not soften with K_L = K",
    "chaboche2": "no sub-stepping at the Gauss points: 1 % off with 10 s steps",
}


def test_expected_failures_are_computed_cases():
    names = {folder.name for folder in CASES}
    assert set(EXPECTED_FAILURES) <= names


@pytest.mark.parametrize("folder", CASES, ids=lambda p: p.name)
def test_case_meets_its_tolerance(folder):
    case = tomllib.loads((folder / "case.toml").read_text(encoding="utf-8"))
    script = _load(folder / "run.py", f"validation_{folder.name.replace('-', '_')}")
    values = script.main(None)
    refs = DOCUMENT.references(case, folder)
    checked, failures = DOCUMENT.check(case, values, refs)
    assert checked, f"{folder.name}: no quantity carries a tolerance"
    if folder.name in EXPECTED_FAILURES:
        assert failures, (
            f"{folder.name} now meets its tolerance: remove it from EXPECTED_FAILURES"
        )
    else:
        assert not failures, failures
