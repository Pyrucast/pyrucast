"""Portique (frame) 2-D — console inclinée, charge perpendiculaire.

Physique
--------
Poutre 2-D orientée à 3 DOFs par nœud (`u_x, u_y, rz`). La rigidité locale
combine l'effort axial (`E·A/L`, comme un treillis), la flexion (`E·I`) et le
cisaillement réduit (`G·A_s`, comme la poutre de Timoshenko), puis est tournée
dans le repère global : `K = Tᵀ K_loc T`, où `T` vient des cosinus directeurs
de l'élément — n'importe quelle orientation dans le plan fonctionne.

Problème
--------
Console inclinée à 45°, encastrée à la base (`u_x = u_y = rz = 0`), charge `P`
**perpendiculaire** à la poutre au bout libre. La charge étant purement
transverse, le bout se déplace de `δ = P·L³/(3·E·I) + P·L/(G·A_s)` le long de la
perpendiculaire (déplacement axial ≈ 0).

Lancement ::

    maturin develop --features extension-module
    python examples/frame.py
"""

import math

import pyrucast

E, A, I, G, A_S, L, P, N = 1.0, 1.0, 1.0, 30.0, 1.0, 1.0, 1.0, 40


def _clamp(node, var, dual):
    imposed = pyrucast.mesher.poi1_from_nodes([node])
    multiplier = pyrucast.mesher.barycenter(imposed)
    return pyrucast.Model.dirichlet(var, dual, imposed, multiplier)


def main() -> None:
    c = s = 1.0 / math.sqrt(2.0)  # direction à 45°
    px, py = -s, c  # perpendiculaire unitaire

    coords = pyrucast.Coords(2)
    base = coords.add_node([0.0, 0.0])
    tip = coords.add_node([L * c, L * s])
    mesh = pyrucast.mesher.line_seg2(base, tip, N)  # ligne de N SEG2 à 45° (`line_seg2`)
    fes = pyrucast.FiniteElementSpace(mesh)

    model = pyrucast.Model.frame(fes)
    for var, dual in (("u_x", "f_x"), ("u_y", "f_y"), ("rz", "m_z")):
        model = model | _clamp(base, var, dual)
    materials = pyrucast.build.material_field(
        model, [("E", E), ("A", A), ("I", I), ("G", G), ("A_s", A_S)]
    )

    load = pyrucast.mesher.poi1_from_nodes([tip])
    rhs = pyrucast.NodeField(load, ["f_x", "f_y"])
    rhs[0].set_value(tip, "f_x", P * px)
    rhs[0].set_value(tip, "f_y", P * py)
    solution = pyrucast.solver.solve(pyrucast.assemble.stiffness(model, materials), rhs)

    delta = P * L**3 / (3.0 * E * I) + P * L / (G * A_S)
    ux = solution.value(tip, "u_x")
    uy = solution.value(tip, "u_y")
    transverse = ux * px + uy * py
    axial = ux * c + uy * s
    print(f"déplacement bout : u = ({ux:.6f}, {uy:.6f})")
    print(f"  transverse = {transverse:.6f}   (analytique δ = {delta:.6f})")
    print(f"  axial      = {axial:.2e}   (≈ 0)")
    assert abs(transverse - delta) < 1e-2 * delta
    assert abs(axial) < 1e-6
    print("OK : déplacement = δ·perpendiculaire, orientation gérée.")


if __name__ == "__main__":
    main()
