"""Exemple du book : récupérer le maillage d'une session gmsh vivante.

Porté au niveau module pour que l'extrait inclus par le book s'affiche en
colonne 0. Exige gmsh, donc marqué et sauté à l'import quand il manque.
"""

import pytest

try:
    import gmsh
except Exception as e:  # noqa: BLE001
    # Pas seulement `ImportError` : la roue gmsh charge son `libgmsh.so` par
    # ctypes et lève `OSError` si les bibliothèques OpenGL du système manquent.
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

# — pyrucast vient chercher le résultat, sans passer par un fichier —
coords = pyrucast.Coords(dim=3)
regions = pyrucast.mesh.from_gmsh(coords)

piece = regions["piece"]
print(piece.element_types())  # ['TET4']
print(regions["encastrement"].element_types())  # ['TRI3']
print(coords.node_count())  # les nœuds du modèle, partagés par les deux

gmsh.finalize()  # pyrucast possède ses données : le maillage lui survit
# ANCHOR_END: from_gmsh

assert piece.element_types() == ["TET4"]
assert regions["encastrement"].element_types() == ["TRI3"]
assert set(regions) == {"encastrement", "piece", "<ungrouped>"}
assert coords.node_count() > 0
