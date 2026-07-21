"""Élasticité linéaire — traction d'un carré (contraintes planes).

Physique
--------
Continuum en petites déformations : équilibre `∇·σ = 0`, loi de Hooke
`σ = D : ε`, cinématique `ε = ½(∇u + ∇uᵀ)`. La rigidité est
`K = ∫ Bᵀ D B dΩ` (B : matrice déformation-déplacement en Voigt, D : matrice
constitutive isotrope, ici en contraintes planes).

Problème
--------
Carré unité, appuis `u_x = 0` (bord gauche) et `u_y = 0` (bord bas), traction
`S` sur le bord droit appliquée en charges nodales cohérentes par l'opérateur
`flux` (sur la composante `f_x`). Solution exacte (uniaxiale) :
`u_x = (S/E)·x`, `u_y = -(ν·S/E)·y`.

Lancement ::

    maturin develop --features extension-module
    python examples/elasticity.py
"""

import pyrucast

E, NU, S, N = 210.0, 0.3, 2.0, 2


def _clamp(nodes, var, dual):
    imposed = pyrucast.mesher.poi1_from_nodes(nodes)
    multiplier = pyrucast.mesher.barycenter(imposed)
    return pyrucast.Model.dirichlet(var, dual, imposed, multiplier)


def main() -> None:
    h = 1.0 / N
    c = pyrucast.Coords(2)

    def idx(i, j):
        return j * (N + 1) + i

    # Grille N×N de QUA4 par balayage de deux lignes SEG2 (`sweep`).
    bottom = pyrucast.mesher.line(c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0]), N)
    top = pyrucast.mesher.line(c.add_node([0.0, 1.0]), c.add_node([1.0, 1.0]), N)
    mesh = pyrucast.mesher.sweep(bottom, top, N)

    # Nœuds rangés par idx(i, j) (i selon x, j selon y) en relisant la
    # connectivité QUA4 : maille (cy, cx) = cy*N + cx, nœuds locaux 0..3.
    grid = [None] * ((N + 1) * (N + 1))
    for cy in range(N):
        for cx in range(N):
            cell = cy * N + cx
            grid[idx(cx, cy)] = mesh.node(0, cell, 0)
            grid[idx(cx + 1, cy)] = mesh.node(0, cell, 1)
            grid[idx(cx + 1, cy + 1)] = mesh.node(0, cell, 2)
            grid[idx(cx, cy + 1)] = mesh.node(0, cell, 3)
    fes = pyrucast.FiniteElementSpace(mesh)

    left = [grid[idx(0, j)] for j in range(N + 1)]
    bottom = [grid[idx(i, 0)] for i in range(N + 1)]
    model = pyrucast.Model.elasticity(fes, "plane_stress")
    model = model | _clamp(left, "u_x", "f_x")
    model = model | _clamp(bottom, "u_y", "f_y")

    materials = pyrucast.build.material_field(model, [("E", E), ("nu", NU)])

    # Traction S sur le bord droit → charges nodales cohérentes (op flux).
    right = pyrucast.Mesh(c, "SEG2")
    for j in range(N):
        right.unit().add_cell([grid[idx(N, j)], grid[idx(N, j + 1)]])
    right_fes = pyrucast.FiniteElementSpace(right)
    rhs = pyrucast.assemble.flux(right_fes[0], S, "f_x")

    solution = pyrucast.solver.solve(pyrucast.assemble.stiffness(model, materials), rhs)

    print(f"{'x':>5} {'y':>5} {'u_x':>12} {'u_y':>12}")
    tol = 1e-10
    for j in range(N + 1):
        for i in range(N + 1):
            x, y = i * h, j * h
            ux = solution.value(grid[idx(i, j)], "u_x")
            uy = solution.value(grid[idx(i, j)], "u_y")
            print(f"{x:5.2f} {y:5.2f} {ux:12.6e} {uy:12.6e}")
            assert abs(ux - S / E * x) < tol
            assert abs(uy + NU * S / E * y) < tol
    print("\nOK : champ uniaxial conforme à u_x=(S/E)x, u_y=-(νS/E)y.")


if __name__ == "__main__":
    main()
