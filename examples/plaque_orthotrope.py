"""Orthotropic elasticity — a plate pulled off its material axes.

Problème
--------
A unit square in plane stress, pulled uniformly (traction ``S``) on its right
edge, with roller supports on the left and bottom edges. The material is
**orthotropic**: stiff in one direction, compliant in the other.

What the example shows is the effect of the **orthotropy frame**. It is given
by vectors, as in Cast3M (``MATE 'DIRECTION' V1 V2``): the components ``V1X``,
``V1Y`` travel in the material field just like the moduli. The first material
axis's angle is swept from 0° to 90°.

Two cases have an analytical solution, and they are the sweep's bounds:

  * **0°** — the stiff axis is aligned with the traction ::

        u_x(1, y) = S / E_1

  * **90°** — it is the compliant axis that works ::

        u_x(1, y) = S / E_2

In between, the plate **shears**: off its axes, an orthotropic material couples
traction and distortion (the ``D_16`` term of the rotated tensor is no longer
zero), and the right edge does not stay straight. That is precisely what
anisotropy brings, and what an isotropic computation cannot produce.

Lancement
---------
Once the extension is built in the venv ::

    maturin develop --features extension-module
    python examples/plaque_orthotrope.py
"""

import math

import pyrucast

# ── Problem data ────────────────────────────────────────────────────────────
E1 = 200.0  # modulus in the stiff direction (material axis 1)
E2 = 50.0  # module transverse
NU12 = 0.25
G12 = 30.0
S = 2.0  # traction on the right edge
N = 4  # grille N×N de QUA4


def maillage():
    """The unit square's QUA4 grid, its nodes and its FE space."""
    h = 1.0 / N
    c = pyrucast.Coords(2)
    grid = [[c.add_node([i * h, j * h]) for i in range(N + 1)] for j in range(N + 1)]
    mesh = pyrucast.Mesh(c, "QUA4")
    for j in range(N):
        for i in range(N):
            mesh.unit().add_cell(
                [grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]]
            )
    return c, grid, pyrucast.FiniteElementSpace(mesh)


def rouleau(target, c, noeuds, variable):
    """A roller support ``variable = 0`` on the given nodes."""
    imposed = pyrucast.Mesh(c, "POI1")
    for n in noeuds:
        imposed.unit().add_cell([n])
    multiplier = pyrucast.mesh.barycenter(imposed)
    return pyrucast.model.dirichlet(target, variable, imposed, multiplier)


def resoudre(angle_deg):
    """The displacement of corner (1, 0) for a material axis at ``angle_deg``."""
    c, grid, fes = maillage()

    # Orthotropic elasticity + both supports.
    model = pyrucast.model.elasticity(fes, "plane_stress", symmetry="orthotropic")
    model = model | rouleau(model, c, [grid[j][0] for j in range(N + 1)], "u_x")
    model = model | rouleau(model, c, [grid[0][i] for i in range(N + 1)], "u_y")

    # The material frame is material data like any other.
    a = math.radians(angle_deg)
    # Traction S on the right edge, as consistent nodal loads: a term of the
    # model, whose density lives in the material.
    bord = pyrucast.Mesh(c, "SEG2")
    for j in range(N):
        bord.unit().add_cell([grid[j][N], grid[j + 1][N]])
    bord_fes = pyrucast.FiniteElementSpace(bord)
    model = model | pyrucast.model.flux(bord_fes, model, "f_x")

    materials = pyrucast.element_field.material_field(
        model,
        [
            ("E_1", E1),
            ("E_2", E2),
            ("E_3", E2),
            ("nu_12", NU12),
            ("nu_13", NU12),
            ("nu_23", 0.25),
            ("G_12", G12),
            ("G_13", G12),
            ("G_23", G12),
            ("V1X", math.cos(a)),
            ("V1Y", math.sin(a)),
            ("phi_f_x", S),
        ],
    )
    rhs = pyrucast.node_field.external_forces(model, materials)

    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)
    coin = grid[0][N]  # (1, 0)
    haut = grid[N][N]  # (1, 1)
    return (
        solution.value(coin, "u_x"),
        solution.value(haut, "u_x") - solution.value(coin, "u_x"),
    )


def main() -> None:
    print("Orthotropic elasticity — sweeping the material frame")
    print(f"  E_1 = {E1}, E_2 = {E2}, nu_12 = {NU12}, G_12 = {G12}, traction S = {S}")
    print()
    print("  angle    u_x(1,0)    u_x gap on the right edge")
    print("  " + "-" * 46)
    for angle in (0.0, 22.5, 45.0, 67.5, 90.0):
        ux, distorsion = resoudre(angle)
        print(f"  {angle:5.1f}°  {ux:10.6f}  {distorsion:+14.6f}")

    # Both bounds are analytical: the stiff axis, then the compliant one.
    ux0, _ = resoudre(0.0)
    ux90, _ = resoudre(90.0)
    print()
    print(f"  0°  : {ux0:.6f}  (attendu S/E_1 = {S / E1:.6f})")
    print(f"  90° : {ux90:.6f}  (attendu S/E_2 = {S / E2:.6f})")
    assert abs(ux0 - S / E1) < 1e-10
    assert abs(ux90 - S / E2) < 1e-10

    # Off axis, the traction induces shear — the right edge warps.
    _, distorsion45 = resoudre(45.0)
    assert abs(distorsion45) > 1e-4, "un orthotrope hors axes doit cisailler"
    print()
    print(f"  At 45°, the right edge warps by {distorsion45:+.6f}:")
    print("  this is the traction/shear coupling of off-axis orthotropy.")


if __name__ == "__main__":
    main()
