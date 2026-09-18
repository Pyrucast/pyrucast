"""Cylindre épais sous pression interne — calcul axisymétrique (solution de Lamé).

A solid of **revolution** is meshed in its meridian plane `(r, z)`:
axisymmetric `Coords` (`x = r`, `y = z`) are enough to make **every** integral
carry the `2πr` factor — stiffness, mass, distributed flux, volumes. The
`"axisymmetric"` model adds the only thing that belongs to the mechanics: the
déformation orthoradiale `ε_θθ = u_r / r`.

Problème : cylindre `a ≤ r ≤ b` sous pression interne `p`, en déformations
plane ends (`u_z = 0` at both ends). Lamé's solution:

    σ_rr = A − B/r²,  σ_θθ = A + B/r²,  u_r = (1+ν)/E · [(1−2ν)·A·r + B/r]
    A = p a²/(b²−a²),  B = p a² b²/(b²−a²)

Lancer :  python examples/axisymetrique_cylindre.py
"""

import pyrucast

E, NU, P = 210_000.0, 0.3, 100.0  # module, Poisson, pression interne
A, B, H = 1.0, 2.0, 0.5  # rayons interne / externe, hauteur
NR, NZ = 40, 1  # mailles radiales / axiales


def main():
    # ── QUA4 mesh of the meridian plane ────────────────────────────────────
    # `Coords.axisymmetric()` : dim 2 implicite, x = r ≥ 0, y = z.
    c = pyrucast.Coords.axisymmetric()

    def idx(i, j):
        return j * (NR + 1) + i

    grid = [
        c.add_node([A + (B - A) * i / NR, H * j / NZ])
        for j in range(NZ + 1)
        for i in range(NR + 1)
    ]
    mesh = pyrucast.Mesh(c, "QUA4")
    for j in range(NZ):
        for i in range(NR):
            mesh.unit().add_cell(
                [
                    grid[idx(i, j)],
                    grid[idx(i + 1, j)],
                    grid[idx(i + 1, j + 1)],
                    grid[idx(i, j + 1)],
                ]
            )
    fes = pyrucast.FiniteElementSpace(mesh)

    # ── Modèle : élasticité axisymétrique + déformations planes ────────────
    # u_z = 0 on both z faces: that is what realizes the plane assumption.
    ends = [grid[idx(i, j)] for i in range(NR + 1) for j in (0, NZ)]
    imposed = pyrucast.mesh.poi1_from_nodes(ends)
    model = pyrucast.model.elasticity(fes, "axisymmetric")
    model = model | pyrucast.model.dirichlet(
        model, "u_y", imposed, pyrucast.mesh.barycenter(imposed)
    )

    # ── Chargement : pression interne sur r = a ────────────────────────────
    # La géométrie étant axisymétrique, `flux` intègre ∫ 2πr N p et donne
    # the total force on the ring directly — no factor by hand.
    inner = pyrucast.Mesh(c, "SEG2")
    for j in range(NZ):
        inner.unit().add_cell([grid[idx(0, j)], grid[idx(0, j + 1)]])
    model = model | pyrucast.model.flux(
        pyrucast.FiniteElementSpace(inner), model, "f_x"
    )
    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("nu", NU), ("phi_f_x", P)]
    )
    rhs = pyrucast.node_field.external_forces(model, materials)

    # ── Assemblage + résolution ────────────────────────────────────────────
    k = pyrucast.matrix.stiffness(model, materials)
    solution = pyrucast.solver.solve(k, rhs)

    # ── Compared with Lamé ─────────────────────────────────────────────────
    a2, b2 = A * A, B * B
    ca = P * a2 / (b2 - a2)
    cb = P * a2 * b2 / (b2 - a2)

    print(f"Cylindre épais {A} ≤ r ≤ {B}, pression interne p = {P}")
    print(f"{'r':>8} {'u_r calculé':>14} {'u_r Lamé':>14} {'écart rel.':>12}")
    worst = 0.0
    for i in range(0, NR + 1, NR // 8):
        n = grid[idx(i, 0)]
        r = A + (B - A) * i / NR
        got = solution.value(n, "u_x")
        exact = (1.0 + NU) / E * ((1.0 - 2.0 * NU) * ca * r + cb / r)
        rel = abs(got - exact) / abs(exact)
        worst = max(worst, rel)
        print(f"{r:8.4f} {got:14.6e} {exact:14.6e} {rel:11.2%}")
    print(f"\nLargest relative gap on the displacement: {worst:.2%}")

    # The part's volume comes out of the same geometry, with no added factor.
    ones = pyrucast.NodeField(mesh, ["one"])
    ones.add_to_component("one", 1.0)
    volume = pyrucast.measure.integral(ones, "one", fes)
    print(
        f"Volume de révolution : {volume:.6f} (exact : {3.141592653589793 * (b2 - a2) * H:.6f})"
    )


if __name__ == "__main__":
    main()
