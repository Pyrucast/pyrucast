"""Élasticité orthotrope — plaque tirée hors de ses axes matériau.

Problème
--------
Un carré unité en contraintes planes, tiré uniformément (traction ``S``) sur son
bord droit, avec des appuis à rouleaux sur les bords gauche et bas. Le matériau
est **orthotrope** : rigide dans une direction, souple dans l'autre.

Ce que l'exemple montre, c'est l'effet du **repère d'orthotropie**. Il est donné
par des vecteurs, comme dans Cast3M (``MATE 'DIRECTION' V1 V2``) : les
composantes ``V1X``, ``V1Y`` voyagent dans le champ matériau au même titre que
les modules. On balaie l'angle du premier axe matériau de 0° à 90°.

Deux cas ont une solution analytique, et ce sont les bornes du balayage :

  * **0°** — l'axe rigide est aligné sur la traction ::

        u_x(1, y) = S / E_1

  * **90°** — c'est l'axe souple qui travaille ::

        u_x(1, y) = S / E_2

Entre les deux, la plaque **cisaille** : hors de ses axes, un matériau
orthotrope couple traction et distorsion (le terme ``D_16`` du tenseur tourné
n'est plus nul), et le bord droit ne reste pas droit. C'est précisément ce que
l'anisotropie apporte, et ce qu'un calcul isotrope ne peut pas produire.

Lancement
---------
Après avoir compilé l'extension dans le venv ::

    maturin develop --features extension-module
    python examples/plaque_orthotrope.py
"""

import math

import pyrucast

# ── Données du problème ──────────────────────────────────────────────────────
E1 = 200.0  # module dans la direction rigide (axe matériau 1)
E2 = 50.0  # module transverse
NU12 = 0.25
G12 = 30.0
S = 2.0  # traction sur le bord droit
N = 4  # grille N×N de QUA4


def maillage():
    """La grille QUA4 du carré unité, ses nœuds et son espace EF."""
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


def rouleau(c, noeuds, variable, dual):
    """Appui glissant ``variable = 0`` sur les nœuds donnés."""
    imposed = pyrucast.Mesh(c, "POI1")
    for n in noeuds:
        imposed.unit().add_cell([n])
    multiplier = pyrucast.mesh.barycenter(imposed)
    return pyrucast.model.dirichlet(variable, dual, imposed, multiplier)


def resoudre(angle_deg):
    """Le déplacement du coin (1, 0) pour un axe matériau à ``angle_deg``."""
    c, grid, fes = maillage()

    # Élasticité orthotrope + les deux appuis.
    model = pyrucast.model.elasticity(fes, "plane_stress", symmetry="orthotropic")
    model = model | rouleau(c, [grid[j][0] for j in range(N + 1)], "u_x", "f_x")
    model = model | rouleau(c, [grid[0][i] for i in range(N + 1)], "u_y", "f_y")

    # Le repère matériau est une donnée matériau comme une autre.
    a = math.radians(angle_deg)
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
        ],
    )

    # Traction S sur le bord droit, en charges nodales cohérentes.
    bord = pyrucast.Mesh(c, "SEG2")
    for j in range(N):
        bord.unit().add_cell([grid[j][N], grid[j + 1][N]])
    bord_fes = pyrucast.FiniteElementSpace(bord)
    rhs = pyrucast.node_field.flux(bord_fes, S, "f_x")

    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)
    coin = grid[0][N]  # (1, 0)
    haut = grid[N][N]  # (1, 1)
    return (
        solution.value(coin, "u_x"),
        solution.value(haut, "u_x") - solution.value(coin, "u_x"),
    )


def main() -> None:
    print("Élasticité orthotrope — balayage du repère matériau")
    print(f"  E_1 = {E1}, E_2 = {E2}, nu_12 = {NU12}, G_12 = {G12}, traction S = {S}")
    print()
    print("  angle    u_x(1,0)    écart u_x sur le bord droit")
    print("  " + "-" * 46)
    for angle in (0.0, 22.5, 45.0, 67.5, 90.0):
        ux, distorsion = resoudre(angle)
        print(f"  {angle:5.1f}°  {ux:10.6f}  {distorsion:+14.6f}")

    # Les deux bornes sont analytiques : l'axe rigide, puis l'axe souple.
    ux0, _ = resoudre(0.0)
    ux90, _ = resoudre(90.0)
    print()
    print(f"  0°  : {ux0:.6f}  (attendu S/E_1 = {S / E1:.6f})")
    print(f"  90° : {ux90:.6f}  (attendu S/E_2 = {S / E2:.6f})")
    assert abs(ux0 - S / E1) < 1e-10
    assert abs(ux90 - S / E2) < 1e-10

    # Hors axes, la traction induit du cisaillement — le bord droit se gauchit.
    _, distorsion45 = resoudre(45.0)
    assert abs(distorsion45) > 1e-4, "un orthotrope hors axes doit cisailler"
    print()
    print(f"  À 45°, le bord droit se gauchit de {distorsion45:+.6f} :")
    print("  c'est le couplage traction/cisaillement de l'orthotropie hors axes.")


if __name__ == "__main__":
    main()
