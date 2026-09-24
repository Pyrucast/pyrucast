"""Book example: fetching the mesh of a live gmsh session.

Put at module level so the excerpt the book includes shows up at
column 0. Requires gmsh, hence marked and skipped at import when it is missing.
"""

import os
import tempfile

import pytest

try:
    import gmsh
except Exception as e:  # noqa: BLE001
    # Not only `ImportError`: the gmsh wheel loads its `libgmsh.so` through
    # ctypes and raises `OSError` if the system's OpenGL libraries are missing.
    pytest.skip(f"gmsh indisponible : {e}", allow_module_level=True)

pytestmark = pytest.mark.gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

# ANCHOR: from_gmsh
import pyrucast

# — la géométrie et le maillage restent l'affaire de gmsh —
gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2, [1], name="encastrement")
gmsh.model.addPhysicalGroup(3, [1], name="piece")
gmsh.model.mesh.generate(3)

# — pyrucast comes and fetches the result, without going through a file —
coords = pyrucast.Coords(dim=3)
regions, _ = pyrucast.mesh.from_gmsh(coords)

piece = regions["piece"]
print(piece.element_types())  # ['TET4']
print(regions["encastrement"].element_types())  # ['TRI3']
print(coords.node_count())  # the model's nodes, shared by both

gmsh.finalize()  # pyrucast owns its data: the mesh outlives it
# ANCHOR_END: from_gmsh

assert piece.element_types() == ["TET4"]
assert regions["encastrement"].element_types() == ["TRI3"]
assert set(regions) == {"encastrement", "piece", "<ungrouped>"}
assert coords.node_count() > 0

# ── And back: pyrucast's results as gmsh views ──────────────────────────────

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
_CWD = os.getcwd()
os.chdir(tempfile.mkdtemp())
temperature = pyrucast.NodeField(piece, ["T"])
for _z in range(len(temperature)):
    for _i in range(temperature[_z].node_count()):
        temperature[_z].set(_i, 0, float(_i))

# ANCHOR: to_gmsh
# A new gmsh model: the meshes as physical groups, the fields as views.
pyrucast.export.to_gmsh({"piece": piece}, {"T": temperature}, model_name="calcul")
gmsh.write("calcul.msh")  # or gmsh.fltk.run() to look at it
gmsh.view.write(gmsh.view.getTags()[0], "temperature.pos")

# from_gmsh reads the views back: a NodeData view is a NodeField.
_, champs = pyrucast.mesh.from_gmsh(pyrucast.Coords(dim=3))
print(sorted(champs))  # ['T']
# ANCHOR_END: to_gmsh

gmsh.finalize()
os.chdir(_CWD)

assert sorted(champs) == ["T"]
