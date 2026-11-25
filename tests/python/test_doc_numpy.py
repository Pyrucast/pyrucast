"""Book example: reading field zones with numpy.

Put at module level so the excerpts the book includes show up at column 0.
Requires numpy, hence skipped at import when it is missing.
"""

import pytest

np = pytest.importorskip("numpy")

import pyrucast

# Four nodes on a line, and a POI1 zone on them.
_coords = pyrucast.Coords(2)
_nodes = [_coords.add_node([float(i), 0.0]) for i in range(4)]
maillage = pyrucast.mesh.poi1_from_nodes(_nodes)

# A square cut into two triangles, and its finite-element space.
_c = pyrucast.Coords(2)
_s = [_c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0])]
_carre = pyrucast.Mesh(_c, "TRI3")
_carre.unit().add_cell([_s[0], _s[1], _s[2]])
_carre.unit().add_cell([_s[0], _s[2], _s[3]])
fes = pyrucast.FiniteElementSpace(_carre)

# ANCHOR: numpy_lire_noeuds
u = pyrucast.NodeField(maillage, [("UX", 0.3), ("UY", -0.4)])
zone = u[0]
a = np.asarray(zone)  # lu sans copie, en lecture seule
print(a.shape, zone.components())  # (4, 2) ['UX', 'UY']
norme = np.linalg.norm(a, axis=1)  # numpy calcule sur la mémoire du champ
print(norme)  # [0.5 0.5 0.5 0.5]
# ANCHOR_END: numpy_lire_noeuds
assert a.shape == (4, 2)
assert np.allclose(norme, 0.5)

# ANCHOR: numpy_lire_gauss
materiau = pyrucast.ElementField(fes, [("E", 210e3), ("NU", 0.3)])
m = np.asarray(materiau[0])  # (mailles, points de Gauss, composantes)
print(m.shape)  # (2, 3, 2) : 2 triangles, 3 points de Gauss, E et NU
print(m[:, :, 0].mean(axis=1))  # E moyen de chaque maille : [210000. 210000.]
# ANCHOR_END: numpy_lire_gauss
assert m.shape == (2, materiau[0].gauss_count, 2)

# ANCHOR: numpy_pret
try:
    zone.set_uniform("UX", 1.0)  # le tableau `a` lit la zone
except BufferError as erreur:
    print(erreur)  # object is read by 1 buffer view(s) …: release them before writing
del a  # le dernier tableau sur la zone disparaît
zone.set_uniform("UX", 1.0)  # l'écriture passe
# ANCHOR_END: numpy_pret
assert zone.max("UX") == 1.0
