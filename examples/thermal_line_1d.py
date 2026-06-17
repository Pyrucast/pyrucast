"""Conduction thermique 1-D — ligne chauffée, comparée à l'analytique.

Problème
--------
Sur le segment [0, 1] :

  * en x = 0 : une **source de chaleur** (flux de Neumann ``Q``) ;
  * en x = 1 : une **température imposée** ``T = 20`` (Dirichlet).

En régime stationnaire sans génération volumique, ``T'' = 0`` : le profil est
linéaire. La solution analytique est ::

    u(x) = 20 + (Q / k) * (1 - x)

et le multiplicateur de Lagrange au nœud imposé (la *réaction* qui maintient
``T = 20``) vaut exactement ``Q`` : tout le flux injecté en x = 0 ressort en
x = 1 (bilan d'énergie).

C'est l'équivalent Python du test d'intégration Rust ``tests/thermal_line.rs``
(et du chapitre « Conduction thermique » du livre).

Lancement
---------
Après avoir compilé l'extension dans le venv ::

    maturin develop --features extension-module
    python examples/thermal_line_1d.py
"""

import pyrucast

# ── Données du problème ──────────────────────────────────────────────────────
K = 1.0          # conductivité
Q = 10.0         # source de chaleur (flux de Neumann) en x = 0
T_IMPOSED = 20.0  # température imposée en x = 1
N_ELEMS = 4


def main() -> None:
    h = 1.0 / N_ELEMS

    # ── Maillage : une ligne de SEG2 sur [0, 1] ──────────────────────────────
    c = pyrucast.Coords(1)
    nodes = [c.add_node([i * h]) for i in range(N_ELEMS + 1)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for i in range(N_ELEMS):
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
    fes = pyrucast.FiniteElementSpace(mesh)

    # ── Modèle : conduction + Dirichlet T = 20 en x = 1 ──────────────────────
    # Le support des multiplicateurs est fabriqué depuis le nœud imposé par le
    # mesher `barycenter` (un nœud neuf colocalisé). Le modèle ne crée rien.
    imposed = pyrucast.poi1_from_nodes([nodes[-1]])
    multiplier = pyrucast.barycenter(imposed)
    mult = multiplier.node(0, 0, 0)
    model = pyrucast.Model.heat_conduction(fes) | pyrucast.Model.dirichlet(
        "T", "q", imposed, multiplier
    )

    # ── Matériau : k uniforme (Dirichlet ignoré automatiquement) ─────────────
    materials = pyrucast.material_field(model, [("k", K)])

    # ── Chargement : source Q en x = 0 (composante duale "q"), valeur imposée
    #    T = 20 au nœud-multiplicateur (slot "imposed_T") ─────────────────────
    load_mesh = pyrucast.Mesh(c, "POI1")
    load_mesh.unit().add_cell([nodes[0]])
    load_mesh.unit().add_cell([mult])
    rhs = pyrucast.NodeField(load_mesh, ["imposed_T", "q"])
    rhs[0].set_value(nodes[0], "q", Q)
    rhs[0].set_value(mult, "imposed_T", T_IMPOSED)

    # ── Assemblage + résolution ──────────────────────────────────────────────
    K_mat = pyrucast.stiffness(model, materials)
    solution = pyrucast.solve(K_mat, rhs)

    # ── Comparaison à l'analytique u(x) = 20 + (Q/k)(1 - x) ──────────────────
    print(f"{'x':>6} {'T_calc':>12} {'T_exact':>12}")
    tol = 1e-10
    for i, node in enumerate(nodes):
        x = i * h
        expected = T_IMPOSED + (Q / K) * (1.0 - x)
        got = solution.value(node, "T")
        print(f"{x:6.3f} {got:12.6f} {expected:12.6f}")
        assert abs(got - expected) < tol, f"x={x}: {got} != {expected}"

    # La réaction (multiplicateur) équilibre le flux injecté : λ = Q.
    reaction = solution.value(mult, "lambda_T")
    print(f"\nréaction λ = {reaction:.6f}  (attendu {Q})")
    assert abs(reaction - Q) < tol

    print("\nOK : profil et réaction conformes à la solution analytique.")


if __name__ == "__main__":
    main()
