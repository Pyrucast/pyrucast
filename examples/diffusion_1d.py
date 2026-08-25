"""Diffusion de Fick 1-D — barreau alimenté en espèce, comparé à l'analytique.

Problème
--------
Sur le segment [0, 1] :

  * en x = 0 : un **flux d'espèce** imposé ``J`` (Neumann) ;
  * en x = 1 : une **concentration imposée** ``c = 1`` (Dirichlet).

En régime stationnaire sans source volumique le profil est linéaire ::

    c(x) = 1 + (J / D) * (1 - x)

et le multiplicateur de Lagrange au nœud imposé vaut exactement ``J`` : tout ce
qui entre en x = 0 ressort en x = 1 (bilan de matière).

L'opérateur est celui de la conduction thermique ; ce qui change, c'est la
**physique**. La primale est la concentration ``c``, la duale le flux ``j``, et
la nature déclarée est ``"diffusion"`` — si bien qu'un modèle couplé
thermo-diffusif se sépare avec ``model.filter(...)``, ce que la deuxième partie
de l'exemple montre.

C'est l'équivalent Python du test d'intégration Rust ``tests/fick.rs``.

Lancement
---------
Après avoir compilé l'extension dans le venv ::

    maturin develop --features extension-module
    python examples/diffusion_1d.py
"""

import pyrucast

# ── Données du problème ──────────────────────────────────────────────────────
SPECIES = "H2"  # l'espèce qui diffuse — tous les noms la portent
D = 2.0  # diffusivité
J = 10.0  # flux d'espèce injecté en x = 0
C_IMPOSED = 1.0  # concentration imposée en x = 1
N_ELEMS = 4
K = 5.0  # conductivité thermique, pour la partie couplée


def ligne(n_elems):
    """Une ligne de ``n_elems`` SEG2 sur [0, 1], avec ses nœuds."""
    c = pyrucast.Coords(1)
    h = 1.0 / n_elems
    nodes = [c.add_node([i * h]) for i in range(n_elems + 1)]
    mesh = pyrucast.Mesh(c, "SEG2")
    for i in range(n_elems):
        mesh.unit().add_cell([nodes[i], nodes[i + 1]])
    return c, nodes, pyrucast.FiniteElementSpace(mesh), h


def profil_stationnaire() -> None:
    c, nodes, fes, h = ligne(N_ELEMS)

    # ── Modèle : diffusion + concentration imposée en x = 1 ──────────────────
    imposed = pyrucast.Mesh(c, "POI1")
    imposed.unit().add_cell([nodes[-1]])
    multiplier = pyrucast.mesh.barycenter(imposed)
    mult = multiplier.node(0, 0, 0)

    model = pyrucast.model.fick(fes, SPECIES) | pyrucast.model.dirichlet(
        f"c_{SPECIES}", f"j_{SPECIES}", imposed, multiplier
    )
    materials = pyrucast.element_field.material_field(model, [(f"D_{SPECIES}", D)])

    # ── Chargement : flux J en x = 0, valeur imposée au multiplicateur ───────
    load = pyrucast.Mesh(c, "POI1")
    load.unit().add_cell([nodes[0]])
    load.unit().add_cell([mult])
    rhs = pyrucast.NodeField(load, [f"imposed_c_{SPECIES}", f"j_{SPECIES}"])
    rhs[0].set_value(nodes[0], f"j_{SPECIES}", J)
    rhs[0].set_value(mult, f"imposed_c_{SPECIES}", C_IMPOSED)

    # ── Assemblage + résolution ─────────────────────────────────────────────
    stiffness = pyrucast.matrix.stiffness(model, materials)
    solution = pyrucast.solver.solve(stiffness, rhs)

    print("Diffusion de Fick 1-D")
    print(f"  D = {D}, flux injecté J = {J}, c(1) = {C_IMPOSED}")
    print()
    print("     x      c calculé    c analytique")
    print("  " + "-" * 36)
    for i, node in enumerate(nodes):
        x = i * h
        attendu = C_IMPOSED + (J / D) * (1.0 - x)
        obtenu = solution.value(node, f"c_{SPECIES}")
        print(f"  {x:5.3f}   {obtenu:10.6f}   {attendu:12.6f}")
        assert abs(obtenu - attendu) < 1e-10

    reaction = solution.value(mult, f"lambda_c_{SPECIES}")
    print()
    print(f"  Bilan de matière : réaction = {reaction:.6f}, flux injecté = {J}")
    assert abs(reaction - J) < 1e-10


def couplage_avec_la_thermique() -> None:
    """Diffusion et conduction sur le même maillage : deux physiques distinctes."""
    _c, _nodes, fes, _h = ligne(3)
    model = pyrucast.model.fick(fes, SPECIES) | pyrucast.model.heat_conduction(fes)

    # Un seul champ matériau porte les deux jeux : l'assembleur résout chaque
    # zone par les composantes que sa physique exige (`D` ici, `k` là).
    materials = pyrucast.element_field.material_field(
        model, [(f"D_{SPECIES}", D), ("k", K)]
    )
    pyrucast.matrix.stiffness(model, materials)

    print()
    print("Modèle couplé diffusion + thermique")
    print(f"  sous-modèles          : {len(model)}")
    print(f"  filter('diffusion')   : {len(model.filter('diffusion'))}")
    print(f"  filter('thermal')     : {len(model.filter('thermal'))}")
    print(f"  filter('mechanical')  : {len(model.filter('mechanical'))}")
    assert len(model.filter("diffusion")) == 1
    assert len(model.filter("thermal")) == 1
    assert len(model.filter("mechanical")) == 0


def main() -> None:
    profil_stationnaire()
    couplage_avec_la_thermique()


if __name__ == "__main__":
    main()
