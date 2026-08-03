"""Poutre de Timoshenko — console, convergence sans verrouillage.

Physique
--------
Poutre planaire déformable en cisaillement. Cinématique : courbure `κ = θ'`,
distorsion `γ = w' - θ`. Efforts : moment `M = E·I·θ'`, effort tranchant
`V = G·A_s·(w' - θ)`. Équilibre : `dV/dx + q = 0`, `dM/dx - V = 0`. Rigidité
`K = ∫ E·I (θ')² dx + ∫ G·A_s (w' - θ)² dx`, le terme de cisaillement étant
intégré de façon **réduite** (1 point) pour éviter le verrouillage.

Problème
--------
Console élancée encastrée (`w = θ = 0`), charge transverse `P` au bout libre.
Solution analytique : `w = P·L³/(3·E·I) + P·L/(G·A_s)` (flexion + cisaillement).
En raffinant, l'élément à intégration réduite converge vers cette valeur (un
élément qui verrouille donnerait une flèche bien trop faible).

Lancement ::

    maturin develop --features extension-module
    python examples/timoshenko.py
"""

import pyrucast

E, I, G, A_S, L, P = 1.0, 1.0, 30.0, 1.0, 1.0, 1.0


def _clamp(node, var, dual):
    imposed = pyrucast.Mesh.poi1_from_nodes([node])
    multiplier = pyrucast.mesh.barycenter(imposed)
    return pyrucast.Model.dirichlet(var, dual, imposed, multiplier)


def tip_deflection(n_elems: int) -> float:
    c = pyrucast.Coords(1)
    base = c.add_node([0.0])
    tip = c.add_node([L])
    mesh = pyrucast.mesh.line(base, tip, n_elems)  # console 1-D (`line`)
    fes = pyrucast.FiniteElementSpace(mesh)

    model = pyrucast.Model.timoshenko(fes)
    model = model | _clamp(base, "w", "f_w")
    model = model | _clamp(base, "theta", "m_theta")

    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("I", I), ("G", G), ("A_s", A_S)]
    )

    load = pyrucast.Mesh.poi1_from_nodes([tip])
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
    assert abs(tip_deflection(40) - analytical) < 1e-2 * analytical
    print("OK : convergence vers la solution Timoshenko (pas de verrouillage).")


if __name__ == "__main__":
    main()
