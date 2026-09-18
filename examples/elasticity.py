"""Élasticité linéaire — traction d'un carré (contraintes planes).

Physique
--------
Continuum en petites déformations : équilibre `∇·σ = 0`, loi de Hooke
`σ = D : ε`, kinematics `ε = ½(∇u + ∇uᵀ)`. The stiffness is
`K = ∫ Bᵀ D B dΩ` (B : matrice déformation-déplacement en Voigt, D : matrice
constitutive isotrope, ici en contraintes planes).

Problème
--------
Carré unité, appuis `u_x = 0` (bord gauche) et `u_y = 0` (bord bas), traction
`S` on the right edge applied as consistent nodal loads by the `flux` operator
(on the `f_x` component). Exact (uniaxial) solution:
`u_x = (S/E)·x`, `u_y = -(ν·S/E)·y`.

Lancement ::

    maturin develop --features extension-module
    python examples/elasticity.py
"""

import pyrucast

E, NU, S, N = 210.0, 0.3, 2.0, 2


def _clamp(target, nodes, var):
    imposed = pyrucast.mesh.poi1_from_nodes(nodes)
    multiplier = pyrucast.mesh.barycenter(imposed)
    return pyrucast.model.dirichlet(target, var, imposed, multiplier)


def main() -> None:
    h = 1.0 / N
    c = pyrucast.Coords(2)

    def idx(i, j):
        return j * (N + 1) + i

    # An N×N grid of QUA4 by sweeping two SEG2 lines (`sweep`).
    bottom = pyrucast.mesh.line(c.add_node([0.0, 0.0]), c.add_node([1.0, 0.0]), N)
    top = pyrucast.mesh.line(c.add_node([0.0, 1.0]), c.add_node([1.0, 1.0]), N)
    mesh = pyrucast.mesh.sweep(bottom, top, N)

    # Nodes laid out by idx(i, j) (i along x, j along y) by reading the
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
    model = pyrucast.model.elasticity(fes, "plane_stress")
    model = model | _clamp(model, left, "u_x")
    model = model | _clamp(model, bottom, "u_y")

    # Traction S on the right edge → consistent nodal loads (the flux op).
    right = pyrucast.Mesh(c, "SEG2")
    for j in range(N):
        right.unit().add_cell([grid[idx(N, j)], grid[idx(N, j + 1)]])
    right_fes = pyrucast.FiniteElementSpace(right)
    model = model | pyrucast.model.flux(right_fes, model, "f_x")
    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("nu", NU), ("phi_f_x", S)]
    )
    rhs = pyrucast.node_field.external_forces(model, materials)

    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)

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
    print("\nOK: uniaxial field matching u_x=(S/E)x, u_y=-(νS/E)y.")


if __name__ == "__main__":
    main()
