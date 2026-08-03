"""Masques par valeur — `mask` et le sucre de comparaison.

`mask` transforme un champ en un **indicateur 0/1 de même structure** (mêmes
zones, même support, mêmes composantes) : ``1.0`` là où la bande de valeurs
tient, ``0.0`` sinon, **composante par composante**. C'est le ``MASQUE`` de
Cast3M. Comme le résultat a exactement la forme de l'entrée, il se multiplie
terme à terme avec elle — l'usage canonique pour « annuler ce qui sort d'une
bande ».

La bande est fixée par quatre bornes de comparaison qui reprennent une pour
une les opérateurs Python : ``ge`` (``>=``), ``gt`` (``>``), ``le`` (``<=``),
``lt`` (``<``). Et le sucre : ``champ >= x`` construit directement le masque.

Lancement
---------
Après avoir compilé l'extension dans le venv ::

    maturin develop --features extension-module
    python examples/field_mask.py
"""

import pyrucast

# Température (°C) le long d'une ligne de 5 nœuds.
TEMPERATURES = [10.0, 25.0, 50.0, 75.0, 90.0]


def _line_field(values, component="T"):
    """NodeField mono-zone POI1 : un nœud par valeur, une composante."""
    c = pyrucast.Coords(1)
    nodes = [c.add_node([float(i)]) for i in range(len(values))]
    mesh = pyrucast.Mesh(c, "POI1")
    for n in nodes:
        mesh.unit().add_cell([n])
    f = pyrucast.NodeField(mesh, [component])
    for n, v in zip(nodes, values):
        f[0].set_value(n, component, v)
    return f, nodes


def _values(field, nodes, component="T"):
    return [field.value(n, component) for n in nodes]


def main() -> None:
    temperature, nodes = _line_field(TEMPERATURES)

    # ── 1. Masque simple : nœuds « chauds » (T >= 50) ────────────────────────
    chauds = temperature.mask(ge=50.0)
    print(f"{'x':>4} {'T':>7} {'chaud':>7}")
    for i, n in enumerate(nodes):
        print(f"{i:4d} {TEMPERATURES[i]:7.1f} {chauds.value(n, 'T'):7.0f}")
    assert _values(chauds, nodes) == [0.0, 0.0, 1.0, 1.0, 1.0]

    # Combien de nœuds chauds ? Le masque étant 0/1, il suffit de sommer.
    n_chauds = sum(_values(chauds, nodes))
    print(f"\nnœuds chauds : {int(n_chauds)} / {len(nodes)}")
    assert n_chauds == 3.0

    # ── 2. Multiplier par le masque : annuler ce qui sort de la bande ─────────
    # On garde la température des seuls nœuds chauds, les autres tombent à 0.
    chaud_seul = temperature * chauds
    print("\nT restreinte aux nœuds chauds :", _values(chaud_seul, nodes))
    assert _values(chaud_seul, nodes) == [0.0, 0.0, 50.0, 75.0, 90.0]

    # ── 3. Sucre de comparaison : `champ >= x` construit le masque ───────────
    # Strictement équivalent au mask() de l'étape 1.
    assert _values(temperature >= 50.0, nodes) == _values(chauds, nodes)
    # Le raccourci le plus lisible pour « annuler hors bande » :
    chaud_seul_bis = temperature * (temperature >= 50.0)
    assert _values(chaud_seul_bis, nodes) == _values(chaud_seul, nodes)

    # ── 4. Bornes strictes vs inclusives ─────────────────────────────────────
    # Bande ouverte 10 < T < 90 (gt / lt) : exclut les deux extrémités.
    milieu = temperature.mask(gt=10.0, lt=90.0)
    print("\n10 < T < 90 (strict) :", _values(milieu, nodes))
    assert _values(milieu, nodes) == [0.0, 1.0, 1.0, 1.0, 0.0]
    # Avec bornes inclusives (ge / le), les extrémités passent.
    assert _values(temperature.mask(ge=10.0, le=90.0), nodes) == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]

    # ── 5. Masque par composante (le filtre `components`) ────────────────────
    # Un champ de déplacement à deux composantes (UX, UY) ; on ne masque
    # que UX.
    c = pyrucast.Coords(1)
    vnodes = [c.add_node([float(i)]) for i in range(3)]
    vmesh = pyrucast.Mesh(c, "POI1")
    for n in vnodes:
        vmesh.unit().add_cell([n])
    depl = pyrucast.NodeField(vmesh, ["UX", "UY"])
    for n, ux, uy in zip(vnodes, [1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]):
        depl[0].set_value(n, "UX", ux)
        depl[0].set_value(n, "UY", uy)

    # Masque « positif » sur UX seulement : UY reste à 1.0 (neutre du produit),
    # donc `depl * m` annule UX < 0 mais laisse UY intact.
    m = depl.mask(ge=0.0, components=["UX"])
    filtre = depl * m
    ux = [filtre.value(n, "UX") for n in vnodes]
    uy = [filtre.value(n, "UY") for n in vnodes]
    print("\nUX (négatifs annulés) :", ux)
    print("UY (inchangé)         :", uy)
    assert ux == [1.0, 0.0, 3.0]
    assert uy == [-1.0, 2.0, -3.0]

    print("\nOK : masques et sucre de comparaison conformes.")


if __name__ == "__main__":
    main()
