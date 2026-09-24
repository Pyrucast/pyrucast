"""Book example: exchanging meshes and fields with MED through medcoupling.

Put at module level so the excerpts the book includes show up at column 0.
Requires medcoupling, hence marked and skipped at import when it is missing.
"""

import os
import tempfile

import pytest

mc = pytest.importorskip("medcoupling")

pytestmark = pytest.mark.medcoupling

_CWD = os.getcwd()
os.chdir(tempfile.mkdtemp())

import pyrucast  # noqa: E402

# A plate of two triangles and a quadrangle, its bottom edge, and fields.
_coords = pyrucast.Coords(dim=2)
_meshes, _, _ = pyrucast.mesh.from_arrays(
    _coords,
    [1, 2, 3, 4, 5, 6],
    [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0],
    [
        ("TRI3", [1, 2, 5, 1, 5, 4], ["plaque"]),
        ("QUA4", [2, 3, 6, 5], ["plaque"]),
        ("SEG2", [1, 2, 2, 3], ["bas"]),
    ],
)
plaque = _meshes["plaque"]
bas = _meshes["bas"]
temperature = pyrucast.NodeField(plaque, ["T"])
froid = pyrucast.NodeField(plaque, ["T"])
for _z in range(len(temperature)):
    for _i in range(temperature[_z].node_count()):
        temperature[_z].set(_i, 0, 20.0 + _i)
contraintes = pyrucast.ElementField(pyrucast.FiniteElementSpace(plaque), ["sxx"])

# ANCHOR: to_medcoupling
# Each key becomes a MED group; an Evolution gives one time step per value.
chauffe = pyrucast.Evolution([(0.0, froid), (60.0, temperature)])
donnees = pyrucast.export.to_medcoupling(
    {"plaque": plaque, "bas": bas},
    {"T": chauffe, "sxx": contraintes},  # sxx: ON_GAUSS_PT
    mesh_name="plaque",
)
donnees.write("plaque.med", 2)  # a medcoupling object: its own writer
# ANCHOR_END: to_medcoupling

# ANCHOR: from_medcoupling
coords = pyrucast.Coords(dim=2)
regions, champs = pyrucast.mesh.from_medcoupling(coords, "plaque.med")
print(sorted(regions))  # ['bas', 'plaque'] — one Mesh per MED group
print(type(champs["T"]).__name__)  # Evolution: two time steps
print(champs["T"].shared_abscissas())  # [0.0, 60.0]
print(type(champs["sxx"]).__name__)  # ElementField, on pyrucast's Gauss points
# ANCHOR_END: from_medcoupling

assert sorted(regions) == ["bas", "plaque"]
assert regions["plaque"].cell_counts() == plaque.cell_counts()
assert champs["T"].shared_abscissas() == [0.0, 60.0]
assert isinstance(champs["sxx"], pyrucast.ElementField)
os.chdir(_CWD)
