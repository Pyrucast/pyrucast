"""Évolution comme loi matériau — module d'Young fonction de la température.

Idée
----
Une évolution **scalaire** typée `E(T)` (module d'Young en fonction de la
température) s'utilise comme une **fonction de transfert** : au lieu d'un
scalaire, on lui passe un **champ de température** et elle rend un **champ de
module d'Young**, nœud par nœud.

Deux ingrédients typés :

- `abscissa_type="T"`  — choisit la composante lue dans le champ d'entrée ;
- `ordinate_type="young"` — nomme la composante produite.

La **correspondance de type** est vérifiée : si le champ d'entrée n'a pas de
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

    # Utilisation scalaire classique : E à 150 °C (interpolation linéaire).
    print("E(150 °C) =", law.interpolate(150.0), "Pa")

    # ── Un champ de température sur une ligne de 5 nœuds ──────────────────────
    c = pc.Coords(1)
    nodes = [c.add_node([float(i)]) for i in range(5)]
    mesh = pc.Mesh(c, "POI1")
    for n in nodes:
        mesh.unit().add_cell([n])
    temperature = pc.NodeField(mesh, ["T"])
    for i, n in enumerate(nodes):
        temperature[0].set_value(n, "T", 25.0 * i)  # 0, 25, 50, 75, 100 °C

    # ── Champ → champ : la loi appliquée nœud par nœud ───────────────────────
    young = law.interpolate(temperature)
    print("composante produite :", young.components())  # ['young']
    for i, n in enumerate(nodes):
        t = temperature[0].value(n, "T")
        e = young[0].value(n, "young")
        print(f"  nœud {i}: T = {t:6.1f} °C  →  E = {e / 1e9:6.2f} GPa")


if __name__ == "__main__":
    main()
