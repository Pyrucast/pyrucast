"""Poutre de Timoshenko — console exacte dès un seul élément.

Physique
--------
Poutre déformable en cisaillement. Cinématique : courbure `κ = θ'`, distorsion
`γ = w' - θ`. Efforts : moment `M = E·I·θ'`, effort tranchant
`V = G·A_s·(w' - θ)`. Équilibre : `dV/dx + q = 0`, `dM/dx - V = 0`.

L'élément assemblé est la **solution exacte** de ces deux équations sur une
travée libre d'efforts répartis — la forme fermée paramétrée par
`Φ = 12·E·I/(G·A_s·L²)`. Ses fonctions de forme dépendent donc du matériau, ce
qu'aucun espace éléments finis ne peut tabuler : l'espace déclare
`MODEL_EMBEDDED`, c'est-à-dire que la formulation possède son interpolation.

Problème
--------
Console encastrée (`w = θ = 0`), charge transverse `P` au bout libre. Solution
analytique `w = P·L³/(3·E·I) + P·L/(G·A_s)` — les deux souplesses, flexion et
cisaillement, **en série**.

L'élément étant exact aux nœuds, **un seul** suffit : raffiner ne change rien,
ce que ce script vérifie. (La version précédente était linéaire à cisaillement
sous-intégré ; elle convergeait vers cette valeur au lieu de l'atteindre, et cet
exemple montrait sa convergence.)

Lancement ::

    maturin develop --features extension-module
    python examples/timoshenko.py
"""

import pyrucast

E, I, G, A_S, L, P = 1.0, 1.0, 30.0, 1.0, 1.0, 1.0


def _clamp(node, var, dual):
    imposed = pyrucast.mesh.poi1_from_nodes([node])
    multiplier = pyrucast.mesh.barycenter(imposed)
    return pyrucast.Model.dirichlet(var, dual, imposed, multiplier)


def tip_deflection(n_elems: int) -> float:
    c = pyrucast.Coords(1)
    base = c.add_node([0.0])
    tip = c.add_node([L])
    mesh = pyrucast.mesh.line(base, tip, n_elems)  # console 1-D (`line`)
    # La base appartient à la formulation, pas à l'espace : elle dépend de `Φ`,
    # donc du matériau, et se calcule maille par maille.
    fes = pyrucast.FiniteElementSpace(mesh, interpolation="MODEL_EMBEDDED")

    model = pyrucast.Model.timoshenko(fes)
    model = model | _clamp(base, "w", "f_w")
    model = model | _clamp(base, "theta", "m_theta")

    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("I", I), ("G", G), ("A_s", A_S)]
    )

    load = pyrucast.mesh.poi1_from_nodes([tip])
    rhs = pyrucast.NodeField(load, ["f_w"])
    rhs[0].set_value(tip, "f_w", P)

    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)
    return solution.value(tip, "w")


def main() -> None:
    analytical = P * L**3 / (3.0 * E * I) + P * L / (G * A_S)
    print(f"{'N':>4} {'w_tip':>12} {'err. rel.':>12}")
    for n in (1, 2, 5, 10, 40):
        w = tip_deflection(n)
        print(f"{n:4d} {w:12.6f} {abs(w - analytical) / analytical:12.2e}")
    print(f"\nanalytique  = {analytical:.6f}  (P·L³/3EI + P·L/GA_s)")

    # Exact aux nœuds : un élément donne déjà la réponse, et raffiner ne
    # l'améliore pas — il n'y a rien à améliorer.
    one = tip_deflection(1)
    assert abs(one - analytical) < 1e-12 * analytical, one
    assert abs(tip_deflection(40) - one) < 1e-12 * analytical
    print("OK : exact aux nœuds dès un élément, le raffinement ne change rien.")


if __name__ == "__main__":
    main()
