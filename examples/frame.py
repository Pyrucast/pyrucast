"""Portique (frame) 2-D — console inclinée, charge perpendiculaire.

Physique
--------
An oriented 2-D beam with 3 DOFs per node (`u_x, u_y, rz`). The local
combine l'effort axial (`E·A/L`, comme un treillis), la flexion (`E·I`) et le
reduced shear (`G·A_s`, like the Timoshenko beam), then is rotated into the
global frame: `K = Tᵀ K_loc T`, where `T` comes from the element's direction
cosines — any orientation in the plane works.

Problème
--------
A cantilever inclined at 45°, clamped at the base (`u_x = u_y = rz = 0`), load
`P` **perpendicular** to the beam at the free end. The load being purely
transverse, le bout se déplace de `δ = P·L³/(3·E·I) + P·L/(G·A_s)` le long de la
perpendiculaire (déplacement axial ≈ 0).

Lancement ::

    maturin develop --features extension-module
    python examples/frame.py
"""

import math

import pyrucast

E, A, I, G, A_S, L, P, N = 1.0, 1.0, 1.0, 30.0, 1.0, 1.0, 1.0, 40


def _clamp(target, node, var):
    imposed = pyrucast.mesh.poi1_from_nodes([node])
    multiplier = pyrucast.mesh.barycenter(imposed)
    return pyrucast.model.dirichlet(target, var, imposed, multiplier)


def main() -> None:
    c = s = 1.0 / math.sqrt(2.0)  # direction at 45°
    px, py = -s, c  # perpendiculaire unitaire

    coords = pyrucast.Coords(2)
    base = coords.add_node([0.0, 0.0])
    tip = coords.add_node([L * c, L * s])
    mesh = pyrucast.mesh.line(base, tip, N)  # a line of N SEG2 at 45° (`line`)
    fes = pyrucast.FiniteElementSpace(mesh, interpolation="MODEL_EMBEDDED")

    model = pyrucast.model.timoshenko(fes)
    for var in ("u_x", "u_y", "r_z"):
        model = model | _clamp(model, base, var)
    materials = pyrucast.element_field.material_field(
        model, [("E", E), ("A", A), ("I", I), ("G", G), ("A_s", A_S)]
    )

    load = pyrucast.mesh.poi1_from_nodes([tip])
    rhs = pyrucast.NodeField(load, ["f_x", "f_y"])
    rhs[0].set_value(tip, "f_x", P * px)
    rhs[0].set_value(tip, "f_y", P * py)
    solution = pyrucast.solver.solve(pyrucast.matrix.stiffness(model, materials), rhs)

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
