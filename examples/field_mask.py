"""Masks by value — `mask` and the comparison sugar.

`mask` turns a field into a **0/1 indicator of the same structure** (same
zones, same support, same components): ``1.0`` where the value band holds,
``0.0`` otherwise, **component by component**. It is Cast3M's ``MASQUE``. As
the result has exactly the input's shape, it multiplies term by term with it —
the canonical use for "zeroing out what falls outside a
bande ».

The band is set by four comparison bounds that match the Python operators one
for one: ``ge`` (``>=``), ``gt`` (``>``), ``le`` (``<=``),
``lt`` (``<``). Et le sucre : ``champ >= x`` construit directement le masque.

Lancement
---------
Once the extension is built in the venv ::

    maturin develop --features extension-module
    python examples/field_mask.py
"""

import pyrucast

# Temperature (°C) along a line of 5 nodes.
TEMPERATURES = [10.0, 25.0, 50.0, 75.0, 90.0]


def _line_field(values, component="T"):
    """A single-zone POI1 NodeField: one node per value, one component."""
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

    # How many hot nodes? The mask being 0/1, summing is enough.
    n_chauds = sum(_values(chauds, nodes))
    print(f"\nnœuds chauds : {int(n_chauds)} / {len(nodes)}")
    assert n_chauds == 3.0

    # ── 2. Multiplying by the mask: zeroing out what falls outside the band ──
    # Only the hot nodes' temperature is kept, the others drop to 0.
    chaud_seul = temperature * chauds
    print("\nT restricted to the hot nodes:", _values(chaud_seul, nodes))
    assert _values(chaud_seul, nodes) == [0.0, 0.0, 50.0, 75.0, 90.0]

    # ── 3. Sucre de comparaison : `champ >= x` construit le masque ───────────
    # Strictly equivalent to step 1's mask().
    assert _values(temperature >= 50.0, nodes) == _values(chauds, nodes)
    # The most readable shortcut for "zero out of band":
    chaud_seul_bis = temperature * (temperature >= 50.0)
    assert _values(chaud_seul_bis, nodes) == _values(chaud_seul, nodes)

    # ── 4. Bornes strictes vs inclusives ─────────────────────────────────────
    # Open band 10 < T < 90 (gt / lt): excludes both ends.
    milieu = temperature.mask(gt=10.0, lt=90.0)
    print("\n10 < T < 90 (strict) :", _values(milieu, nodes))
    assert _values(milieu, nodes) == [0.0, 1.0, 1.0, 1.0, 0.0]
    # With inclusive bounds (ge / le), the ends pass.
    assert _values(temperature.mask(ge=10.0, le=90.0), nodes) == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]

    # ── 5. Mask per component (the `components` filter) ──────────────────────
    # A displacement field with two components (UX, UY); only UX is masked.

    c = pyrucast.Coords(1)
    vnodes = [c.add_node([float(i)]) for i in range(3)]
    vmesh = pyrucast.Mesh(c, "POI1")
    for n in vnodes:
        vmesh.unit().add_cell([n])
    depl = pyrucast.NodeField(vmesh, ["UX", "UY"])
    for n, ux, uy in zip(vnodes, [1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]):
        depl[0].set_value(n, "UX", ux)
        depl[0].set_value(n, "UY", uy)

    # A "positive" mask on UX only: UY stays at 1.0 (the product's identity), so
    # `depl * m` zeroes UX < 0 but leaves UY untouched.
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
