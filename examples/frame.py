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
    imposed = pyrucast.poi1_from_nodes([node])
    multiplier = pyrucast.barycenter(imposed)
    return pyrucast.Model.dirichlet(var, dual, imposed, multiplier)


def main() -> None:
    c = s = 1.0 / math.sqrt(2.0)  # direction à 45°
    px, py = -s, c  # perpendiculaire unitaire
    h = L / N

    coords = pyrucast.Coords(2)
    nodes = [coords.add_node([i * h * c, i * h * s]) for i in range(N + 1)]
    mesh = pyrucast.Mesh(coords, "SEG2")
    for i in range(N):
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
    fes = pyrucast.FiniteElementSpace(mesh)

    model = pyrucast.Model.frame(fes)
    for var, dual in (("u_x", "f_x"), ("u_y", "f_y"), ("rz", "m_z")):
        model = model | _clamp(nodes[0], var, dual)
    materials = pyrucast.material_field(
        model, [("E", E), ("A", A), ("I", I), ("G", G), ("A_s", A_S)]
    )

    load = pyrucast.Mesh(coords, "POI1")
    load.unit().add_cell([nodes[-1]])
    rhs = pyrucast.NodeField(load, ["f_x", "f_y"])
    rhs[0].set_value(nodes[-1], "f_x", P * px)
    rhs[0].set_value(nodes[-1], "f_y", P * py)
    solution = pyrucast.solve(pyrucast.stiffness(model, materials), rhs)

    delta = P * L**3 / (3.0 * E * I) + P * L / (G * A_S)
    ux = solution.value(nodes[-1], "u_x")
    uy = solution.value(nodes[-1], "u_y")
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
