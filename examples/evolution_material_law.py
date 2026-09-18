"""Evolution as a material law — Young's modulus as a function of temperature.

Idée
----
A **scalar** evolution typed `E(T)` (Young's modulus as a function of
temperature) is used as a **transfer function**: instead of a scalar, it is
handed a **temperature field** and it returns a **Young's modulus field**, node
by node.

Deux ingrédients typés :

- `abscissa_type="T"`  — picks the component read in the input field;
- `ordinate_type="young"` — nomme la composante produite.

The **type match** is checked: if the input field has no
composante `"T"`, l'appel échoue.

Exécution
---------
    maturin develop --features extension-module
    python examples/evolution_material_law.py
"""

import pyrucast as pc


def main() -> None:
    # ── Loi matériau E(T) : tabulée, interpolée linéairement ─────────────────
    law = pc.Evolution(
        [(0.0, 210e9), (100.0, 200e9), (300.0, 170e9)],
        abscissa_type="T",
        ordinate_type="young",
    )

    # Classic scalar use: E at 150 °C (linear interpolation).
    print("E(150 °C) =", law.interpolate(150.0), "Pa")

    # ── A temperature field on a line of 5 nodes ─────────────────────────────
    c = pc.Coords(1)
    nodes = [c.add_node([float(i)]) for i in range(5)]
    mesh = pc.Mesh(c, "POI1")
    for n in nodes:
        mesh.unit().add_cell([n])
    temperature = pc.NodeField(mesh, ["T"])
    for i, n in enumerate(nodes):
        temperature[0].set_value(n, "T", 25.0 * i)  # 0, 25, 50, 75, 100 °C

    # ── Field → field: the law applied node by node ──────────────────────────
    young = law.interpolate(temperature)
    print("composante produite :", young.components())  # ['young']
    for i, n in enumerate(nodes):
        t = temperature[0].value(n, "T")
        e = young[0].value(n, "young")
        print(f"  nœud {i}: T = {t:6.1f} °C  →  E = {e / 1e9:6.2f} GPa")


if __name__ == "__main__":
    main()
