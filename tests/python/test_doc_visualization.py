"""Source of the Python examples of `book/src/visualization.md`.

The excerpts write files under short names (`piece.svg`, …): the module
switches the current directory to a temporary folder, so that the displayed
excerpt stays exactly what a user would write.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt. pytest therefore runs this file
at **collection** time; an example that breaks is a collection error, with a
full traceback and a non-zero exit code.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import os
import tempfile

import pyrucast

# A throwaway working directory — the excerpts' file names stay short. The
# module **gives back** the current directory at the end: at module level
# there is no fixture, and leaving it moved would trap the other files.
_TMP = tempfile.TemporaryDirectory()
_CWD = os.getcwd()
os.chdir(_TMP.name)


def _triangle():
    coords = pyrucast.Coords(3)
    n = [
        coords.add_node(p) for p in ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    ]
    mesh = pyrucast.Mesh(coords, "TRI3")
    mesh.unit().add_cell(n)
    return coords, mesh, n


def _champ_nodal(noeuds, composantes, valeur=1.0, support=None):
    """Two calls without `support` build two **distinct** clouds: an `Evolution`
    of fields, by contrast, requires the same support at every step."""
    support = support or pyrucast.mesh.poi1_from_nodes(noeuds)
    f = pyrucast.NodeField(support, composantes)
    for i, noeud in enumerate(noeuds):
        for composante in composantes:
            f[0].set_value(noeud, composante, valeur * (i + 1))
    return f


# ── Formats de sortie ───────────────────────────────────────────────────────


# ── formats ────────────────────────────────────────────────

_, mesh, _ = _triangle()
# ANCHOR: formats
mesh.plot(save="piece.svg")  # to version, to publish
mesh.plot(save="piece.svgz")  # to stack by the hundred
# ANCHOR_END: formats

# ── vue et export ──────────────────────────────────────────

# ANCHOR: vue
import pyrucast

coords = pyrucast.Coords(3)
a = coords.add_node([0.0, 0.0, 0.0])
b = coords.add_node([1.0, 0.0, 0.0])
c = coords.add_node([0.0, 1.0, 0.0])

mesh = pyrucast.Mesh(coords, "TRI3")
mesh.unit().add_cell([a, b, c])

# (yaw, pitch, scale) ; save=None ouvre la fenêtre interactive.
mesh.plot(view=(45.0, 35.264, 1.0), save="triangle.svg")
# ANCHOR_END: vue

# ── titre ──────────────────────────────────────────────────

_, mesh, n = _triangle()
t_field = _champ_nodal(n, ["T"], 20.0)
# ANCHOR: titre
mesh.plot(
    save="piece.svg", title="cantilever beam"
)  # caption centred at the SVG's bottom
mesh.plot(save="t.svg", field=t_field, title="temperature")  # combines with field
# mesh.plot(title="ma pièce")  # nomme la fenêtre interactive (bloquant)
# ANCHOR_END: titre

# ── couleur de face ────────────────────────────────────────

coords, _, _ = _triangle()
# ANCHOR: couleur
sm = pyrucast.Mesh(coords, "TRI3")[0]  # view of the single submesh
sm.face_color = (220, 60, 60)
assert sm.face_color == (220, 60, 60)

# The same colour for **every** zone of a mesh, without a loop: the method
# returns the mesh, so it chains.
piece = pyrucast.Mesh(coords, "TRI3").set_face_color((60, 60, 220))
assert all(zone.face_color == (60, 60, 220) for zone in piece)
# ANCHOR_END: couleur

# ── Champs et échelles ──────────────────────────────────────────────────────


# ── champ et echelle ───────────────────────────────────────

_, mesh, n = _triangle()
t_field = _champ_nodal(n, ["T"], 20.0)
u_field = _champ_nodal(n, ["UX", "UY"], 0.5)
fes = pyrucast.FiniteElementSpace(mesh)
flux_field = pyrucast.ElementField(fes, ["q"])
flux_field[0].set_uniform("q", 3.0)
# ANCHOR: champ
# Default component, viridis, automatic scale.
mesh.plot(save="t.svg", field=t_field)

# Composante "UY", colormap "coolwarm", bornes fixées.
mesh.plot(
    save="uy.svg",
    field=u_field,
    component="UY",
    cmap="coolwarm",
    vmin=-1.0,
    vmax=1.0,
)

# Ceiling only set: the floor follows the data's minimum.
mesh.plot(save="t.svg", field=t_field, vmax=100.0)

# Field at the Gauss points: strictly the same call.
mesh.plot(save="flux.svg", field=flux_field)
# ANCHOR_END: champ

# ── Style ───────────────────────────────────────────────────────────────────


# ── wireframe ──────────────────────────────────────────────

_, mesh, n = _triangle()
t_field = _champ_nodal(n, ["T"], 20.0)
# ANCHOR: wireframe
mesh.plot(save="solide.svg")  # peau opaque (défaut)
mesh.plot(save="fil.svg", wireframe=True)  # fil de fer

# Pointless with a field: raises ValueError.
# mesh.plot(save="x.svg", field=t_field, wireframe=True)
# ANCHOR_END: wireframe

# ── Export VTK ──────────────────────────────────────────────────────────────


def _champ_par_elements(mesh, composante, valeur):
    """A uniform `ElementField`, set up without naming `pyrucast` at the caller.

    An `import pyrucast` inside an anchor makes it a local variable: the module
    becomes unusable before the anchor, within the same function.
    """
    f = pyrucast.ElementField(pyrucast.FiniteElementSpace(mesh), [composante])
    f[0].set_uniform(composante, valeur)
    return f


# ── export vtk ─────────────────────────────────────────────

_, mesh, n = _triangle()
temperature = _champ_nodal(n, ["T"], 20.0)
stresses = _champ_par_elements(mesh, "sigma_xx", 1.0)
# ANCHOR: vtk
import pyrucast

# Géométrie seule.
pyrucast.export.export_vtk(mesh, "maillage.vtk")

# Geometry + field at the nodes (POINT_DATA).
pyrucast.export.export_vtk(mesh, "solution.vtk", field=temperature)

# Geometry + field at the Gauss points (CELL_DATA): one value per cell = the
# intra-element mean of the cell's Gauss points.
pyrucast.export.export_vtk(mesh, "contraintes.vtk", field=stresses)
# ANCHOR_END: vtk

# ── VTK binaire et séries temporelles ──────────────────────────────────────

_, mesh, n = _triangle()
support = pyrucast.mesh.poi1_from_nodes(n)
t0, t1 = (_champ_nodal(n, ["T"], v, support) for v in (20.0, 80.0))
# ANCHOR: vtk_binaire
# Same file, numbers written raw (big-endian): smaller, faster to read.
pyrucast.export.export_vtk(mesh, "solution_bin.vtk", field=t1, binary=True)
# ANCHOR_END: vtk_binaire

# ANCHOR: vtk_serie
# An Evolution of fields: one file per time, plus an index for ParaView.
chauffe = pyrucast.Evolution([(0.0, t0), (30.0, t1)])
pyrucast.export.export_vtk(mesh, "chauffe.vtk.series", field=chauffe, binary=True)
# → chauffe_0000.vtk, chauffe_0001.vtk and chauffe.vtk.series:
#   open the .series in ParaView, the time slider plays the two steps.
# The steps themselves, as whole fields:
print(chauffe.shared_abscissas(), len(chauffe.frames()))  # [0.0, 30.0] 2
# ANCHOR_END: vtk_serie

# ── Évolutions ──────────────────────────────────────────────────────────────


# ── evolution plot ─────────────────────────────────────────

_, mesh, n = _triangle()
maillage = mesh
support = pyrucast.mesh.poi1_from_nodes(n)
champ_t0, champ_t1, champ_t2 = (
    _champ_nodal(n, ["T"], v, support) for v in (10.0, 20.0, 5.0)
)
# ANCHOR: evolution
import pyrucast as pc

# Courbe scalaire (variable → valeur).
e = pc.Evolution([(0.0, 10.0), (1.0, 20.0), (2.0, 5.0)])
e.plot(save="courbe.svg", x_label="temps", y_label="T", title="évolution de T")

# Evolution of a field at the nodes: one whole NodeField per time step.
ev = pc.Evolution([(0.0, champ_t0), (1.0, champ_t1), (2.0, champ_t2)])
ev.plot(save="frame.png", frame=2)  # one tabulated value (default: the last)
ev.plot(save="frame_surf.png", mesh=maillage)  # surface rendering on a supplied mesh
# ANCHOR_END: evolution

# ── Axisymétrie : le corps de révolution ────────────────────────────────────


# ── revolve ────────────────────────────────────────────────

# ANCHOR: revolve
import pyrucast

coords = pyrucast.Coords.axisymmetric()  # (r, z), r ≥ 0
section = [coords.add_node(p) for p in ([1.0, 0.0], [2.0, 0.0], [1.0, 1.0])]
mesh = pyrucast.Mesh(coords, "TRI3")
mesh.unit().add_cell(section)
# … computation, then a temperature field at the section's nodes:
t_field = pyrucast.NodeField(pyrucast.mesh.poi1_from_nodes(section), ["T"])

mesh.plot(save="section.svg")  # la section plane (défaut)
mesh.plot(save="piece.svg", revolve=True)  # le corps de révolution complet
mesh.plot(save="coupe.svg", revolve=True, revolve_angle=270.0)  # opened to 270°
mesh.plot(save="t3d.svg", field=t_field, revolve=True)  # field on the body
# ANCHOR_END: revolve

# End of the excerpts: the current directory is given back.
os.chdir(_CWD)
