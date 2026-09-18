"""Intégrale et résultante d'un champ.

Two "field → scalar" reductions, component by component:

- ``integral(field, comp, fespace=…)`` integrates over the support with the
  finite element quadrature, ``∫_Ω f dΩ``. On a ``NodeField`` the nodal values
  are lifted to the Gauss points by the **shape functions** ``N_i``; on an
  ``ElementField`` the values (already at the Gauss points) are integrated
  directly. The resultant of a distributed force *density* is computed there.
- ``field.sum(comp)`` sums the values node by node — the resultant of a field
  of *already nodal* forces (internal forces, reactions…). ``xtx(field)`` gives
  its squared norm ``Σ v²``.

Lancement
---------
Once the extension is built in the venv ::

    maturin develop --features extension-module
    python examples/field_integral.py
"""

import pyrucast

N = 8  # SEG2 elements on [0, 1]


def _line(n_elems):
    """A SEG2 mesh on ``[0, 1]``, its nodes, and the Lagrange-1 FE space."""
    c = pyrucast.Coords(1)
    nodes = [c.add_node([i / n_elems]) for i in range(n_elems + 1)]
    seg = pyrucast.Mesh(c, "SEG2")
    for i in range(n_elems):
        seg.unit().add_cell([nodes[i], nodes[i + 1]])
    return nodes, seg, pyrucast.FiniteElementSpace(seg)


def main() -> None:
    nodes, seg, fes = _line(N)
    pts = pyrucast.mesh.poi1_from_nodes(nodes)  # nodal (POI1) support of the same nodes

    # ── 1. Integral of a *nodal* field (through the shape functions N_i) ─────
    # f ≡ 1  ⇒  ∫₀¹ 1 dx = longueur = 1.
    unite = pyrucast.NodeField(pts, ["f"])
    for n in nodes:
        unite[0].set_value(n, "f", 1.0)
    mesure = pyrucast.measure.integral(unite, "f", fespace=fes)
    print(f"∫ 1 dx          = {mesure:.6f}   (attendu 1.0 = longueur)")
    assert abs(mesure - 1.0) < 1e-12

    # f(x) = x  ⇒  ∫₀¹ x dx = 1/2  (Lagrange-1 intègre le linéaire exactement).
    rampe = pyrucast.NodeField(pts, ["f"])
    for i, n in enumerate(nodes):
        rampe[0].set_value(n, "f", i / N)
    aire = pyrucast.measure.integral(rampe, "f", fespace=fes)
    print(f"∫ x dx          = {aire:.6f}   (attendu 0.5)")
    assert abs(aire - 0.5) < 1e-12

    # ── 2. The same integral, of a field *by element* (values already at Gauss) ─
    # Constant density c ≡ 3 ⇒ ∫₀¹ 3 dx = 3. No fespace: direct quadrature.
    densite = pyrucast.ElementField(fes, ["c"])
    densite[0].set_uniform("c", 3.0)
    total = pyrucast.measure.integral(densite, "c")
    print(f"∫ 3 dx (Gauss)  = {total:.6f}   (attendu 3.0)")
    assert abs(total - 3.0) < 1e-12

    # ── 3. Resultant of a field of *nodal* forces: sum node by node ──────────
    forces = pyrucast.NodeField(pts, ["fx", "fy"])
    for n in nodes:
        forces[0].set_value(n, "fx", 2.0)  # +2 along x at every node
        forces[0].set_value(n, "fy", -1.0)  # -1 along y at every node
    rx, ry = forces.sum("fx"), forces.sum("fy")
    print(f"resultant       = ({rx:.1f}, {ry:.1f})   over {N + 1} nodes")
    assert rx == 2.0 * (N + 1) and ry == -1.0 * (N + 1)

    # Squared norm (e.g. a convergence criterion on a residual).
    norme2 = pyrucast.measure.xtx(forces)
    print(f"‖forces‖²        = {norme2:.1f}")
    assert norme2 == (N + 1) * (2.0**2 + 1.0**2)


if __name__ == "__main__":
    main()
