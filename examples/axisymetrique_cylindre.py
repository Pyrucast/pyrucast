"""Cylindre épais sous pression interne — calcul axisymétrique (solution de Lamé).

Un solide de **révolution** se maille dans son plan méridien `(r, z)` : des
`Coords` axisymétriques (`x = r`, `y = z`) suffisent à faire porter le facteur
`2πr` à **toutes** les intégrales — rigidité, masse, flux réparti, volumes. Le
modèle `"axisymmetric"` y ajoute la seule chose qui relève de la mécanique : la
déformation orthoradiale `ε_θθ = u_r / r`.

Problème : cylindre `a ≤ r ≤ b` sous pression interne `p`, en déformations
planes (`u_z = 0` aux deux extrémités). Solution de Lamé :

    σ_rr = A − B/r²,  σ_θθ = A + B/r²,  u_r = (1+ν)/E · [(1−2ν)·A·r + B/r]
    A = p a²/(b²−a²),  B = p a² b²/(b²−a²)

Lancer :  python examples/axisymetrique_cylindre.py
"""

import pyrucast

E, NU, P = 210_000.0, 0.3, 100.0  # module, Poisson, pression interne
A, B, H = 1.0, 2.0, 0.5  # rayons interne / externe, hauteur
NR, NZ = 40, 1  # mailles radiales / axiales


def main():
    # ── Maillage QUA4 du plan méridien ─────────────────────────────────────
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
    # u_z = 0 sur les deux faces z : c'est ce qui réalise l'hypothèse plane.
    ends = [grid[idx(i, j)] for i in range(NR + 1) for j in (0, NZ)]
    imposed = pyrucast.mesher.poi1_from_nodes(ends)
    model = pyrucast.Model.elasticity(fes, "axisymmetric")
    model = model | pyrucast.Model.dirichlet(
        "u_y", "f_y", imposed, pyrucast.mesher.barycenter(imposed)
    )
    materials = pyrucast.build.material_field(model, [("E", E), ("nu", NU)])

    # ── Chargement : pression interne sur r = a ────────────────────────────
    # La géométrie étant axisymétrique, `flux` intègre ∫ 2πr N p et donne
    # directement l'effort total sur l'anneau — aucun facteur à la main.
    inner = pyrucast.Mesh(c, "SEG2")
    for j in range(NZ):
        inner.unit().add_cell([grid[idx(0, j)], grid[idx(0, j + 1)]])
    rhs = pyrucast.assemble.flux(pyrucast.FiniteElementSpace(inner)[0], P, "f_x")

    # ── Assemblage + résolution ────────────────────────────────────────────
    k = pyrucast.assemble.stiffness(model, materials)
    solution = pyrucast.solver.solve(k, rhs)

    # ── Comparaison à Lamé ─────────────────────────────────────────────────
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
    print(f"\nÉcart relatif maximal sur le déplacement : {worst:.2%}")

    # Le volume de la pièce sort de la même géométrie, sans facteur ajouté.
    ones = pyrucast.NodeField(mesh, ["one"])
    ones.add_to_component("one", 1.0)
    volume = pyrucast.field.integral(ones, "one", fes)
    print(
        f"Volume de révolution : {volume:.6f} (exact : {3.141592653589793 * (b2 - a2) * H:.6f})"
    )


if __name__ == "__main__":
    main()
