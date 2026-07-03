"""Intégrale et résultante d'un champ.

Deux réductions « champ → scalaire », composante par composante :

- ``integral(field, comp, fespace=…)`` intègre sur le support par la quadrature
  éléments finis, ``∫_Ω f dΩ``. Sur un ``NodeField`` les valeurs nodales sont
  relevées aux points de Gauss par les **fonctions de forme** ``N_i`` ; sur un
  ``ElementField`` les valeurs (déjà aux Gauss) sont intégrées directement. On
  y calcule la résultante d'une *densité* de force distribuée.
- ``field.sum(comp)`` somme les valeurs par nœud — la résultante d'un champ de
  forces *déjà nodales* (efforts internes, réactions…). ``xtx(field)`` en donne
  la norme au carré ``Σ v²``.

Lancement
---------
Après avoir compilé l'extension dans le venv ::

    maturin develop --features extension-module
    python examples/field_integral.py
"""

import pyrucast

N = 8  # éléments SEG2 sur [0, 1]


def _line(n_elems):
    """Maillage SEG2 sur ``[0, 1]``, ses nœuds, et l'espace EF Lagrange-1."""
    c = pyrucast.Coords(1)
    nodes = [c.add_node([i / n_elems]) for i in range(n_elems + 1)]
    seg = pyrucast.Mesh(c, "SEG2")
    for i in range(n_elems):
        seg.unit().add_cell([nodes[i], nodes[i + 1]])
    return nodes, seg, pyrucast.FiniteElementSpace(seg)


def main() -> None:
    nodes, seg, fes = _line(N)
    pts = pyrucast.poi1_from_nodes(nodes)  # support nodal (POI1) des mêmes nœuds

    # ── 1. Intégrale d'un champ *nodal* (via les fonctions de forme N_i) ─────
    # f ≡ 1  ⇒  ∫₀¹ 1 dx = longueur = 1.
    unite = pyrucast.NodeField(pts, ["f"])
    for n in nodes:
        unite[0].set_value(n, "f", 1.0)
    mesure = pyrucast.integral(unite, "f", fespace=fes)
    print(f"∫ 1 dx          = {mesure:.6f}   (attendu 1.0 = longueur)")
    assert abs(mesure - 1.0) < 1e-12

    # f(x) = x  ⇒  ∫₀¹ x dx = 1/2  (Lagrange-1 intègre le linéaire exactement).
    rampe = pyrucast.NodeField(pts, ["f"])
    for i, n in enumerate(nodes):
        rampe[0].set_value(n, "f", i / N)
    aire = pyrucast.integral(rampe, "f", fespace=fes)
    print(f"∫ x dx          = {aire:.6f}   (attendu 0.5)")
    assert abs(aire - 0.5) < 1e-12

    # ── 2. Même intégrale, d'un champ *par élément* (valeurs déjà aux Gauss) ─
    # Densité constante c ≡ 3 ⇒ ∫₀¹ 3 dx = 3. Pas de fespace : quadrature directe.
    densite = pyrucast.ElementField(fes, ["c"])
    densite[0].set_uniform("c", 3.0)
    total = pyrucast.integral(densite, "c")
    print(f"∫ 3 dx (Gauss)  = {total:.6f}   (attendu 3.0)")
    assert abs(total - 3.0) < 1e-12

    # ── 3. Résultante d'un champ de forces *nodales* : somme par nœud ────────
    forces = pyrucast.NodeField(pts, ["fx", "fy"])
    for n in nodes:
        forces[0].set_value(n, "fx", 2.0)  # +2 selon x à chaque nœud
        forces[0].set_value(n, "fy", -1.0)  # -1 selon y à chaque nœud
    rx, ry = forces.sum("fx"), forces.sum("fy")
    print(f"résultante      = ({rx:.1f}, {ry:.1f})   sur {N + 1} nœuds")
    assert rx == 2.0 * (N + 1) and ry == -1.0 * (N + 1)

    # Norme au carré (p.ex. critère de convergence sur un résidu).
    norme2 = pyrucast.xtx(forces)
    print(f"‖forces‖²        = {norme2:.1f}")
    assert norme2 == (N + 1) * (2.0**2 + 1.0**2)


if __name__ == "__main__":
    main()
