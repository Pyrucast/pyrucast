"""1-D Fick diffusion — a bar fed with a species, compared with the analytical solution.

Problème
--------
On the segment [0, 1]:

  * en x = 0 : un **flux d'espèce** imposé ``J`` (Neumann) ;
  * at x = 1: an **imposed concentration** ``c = 1`` (Dirichlet).

In the steady regime without a volume source the profile is linear ::

    c(x) = 1 + (J / D) * (1 - x)

and the Lagrange multiplier at the imposed node is exactly ``J``: everything
entering at x = 0 leaves at x = 1 (mass balance).

The operator is the thermal conduction one; what changes is the **physics**.
The primal is the concentration ``c``, the dual the flux ``j``, and the kind
declared is ``"diffusion"`` — so that a coupled thermo-diffusive model splits
with ``model.filter(...)``, which the second part
de l'exemple montre.

This is the Python equivalent of the Rust integration test ``tests/fick.rs``.

Lancement
---------
Once the extension is built in the venv ::

    maturin develop --features extension-module
    python examples/diffusion_1d.py
"""

import pyrucast

# ── Problem data ────────────────────────────────────────────────────────────
SPECIES = "H2"  # the diffusing species — every name carries it
D = 2.0  # diffusivité
J = 10.0  # flux d'espèce injecté en x = 0
C_IMPOSED = 1.0  # concentration imposée en x = 1
N_ELEMS = 4
K = 5.0  # thermal conductivity, for the coupled part


def ligne(n_elems):
    """A line of ``n_elems`` SEG2 on [0, 1], with its nodes."""
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

    cible = pyrucast.model.fick(fes, SPECIES)

    model = cible | pyrucast.model.dirichlet(cible, f"c_{SPECIES}", imposed, multiplier)
    materials = pyrucast.element_field.material_field(model, [(f"D_{SPECIES}", D)])

    # ── Loading: flux J at x = 0, imposed value at the multiplier ────────────
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
    """Diffusion and conduction on the same mesh: two distinct physics."""
    _c, _nodes, fes, _h = ligne(3)
    model = pyrucast.model.fick(fes, SPECIES) | pyrucast.model.heat_conduction(fes)

    # A single material field carries both sets: the assembler resolves each zone
    # through the components its physics requires (`D` here, `k` there).
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
