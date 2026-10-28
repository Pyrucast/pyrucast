"""Source of the examples of `book/src/pas-a-pas.md`.

Every block of the page comes from here through `{{#include …:anchor}}`. The
setup lives outside the anchors. See `book/src/developper/documentation-et-tests.md`.

The code lives at module level, not inside test functions (mdbook does not
strip the indentation of an included excerpt): pytest runs it at collection.
"""

import pyrucast as pc


def _plaque():
    """A 4×2 QUA4 plate on [0, 2]×[0, 1]: conduction and plane-stress
    plasticity on the same mesh, the left edge heated, the plate held on its
    left (in x) and bottom (in y) edges."""
    c = pc.Coords(2)
    grille = [c.add_node([0.5 * i, 0.5 * j]) for j in range(3) for i in range(5)]

    def noeud(i, j):
        return grille[j * 5 + i]

    maillage = pc.Mesh(c, "QUA4")
    for j in range(2):
        for i in range(4):
            maillage.unit().add_cell(
                [noeud(i, j), noeud(i + 1, j), noeud(i + 1, j + 1), noeud(i, j + 1)]
            )
    fes = pc.FiniteElementSpace(maillage)

    def bloquer(cible, noeuds, variable):
        blocage = pc.model.dirichlet(pc.mesh.poi1_from_nodes(noeuds), cible, variable)
        return blocage, blocage.multiplier_mesh()

    gauche = [noeud(0, j) for j in range(3)]
    bas = [noeud(i, 0) for i in range(5)]
    conduction = pc.model.heat_conduction(fes)
    solide = pc.model.plasticity_perfect(fes, "plane_stress")
    chauffe, support_chauffe = bloquer(conduction, gauche, "T")
    modele = (
        conduction
        | solide
        | chauffe
        | bloquer(solide, gauche, "u_x")[0]
        | bloquer(solide, bas, "u_y")[0]
    )
    return maillage, noeud, modele, support_chauffe


maillage, noeud, modele, support_chauffe = _plaque()


def _temperature_imposee(valeur):
    champ = pc.NodeField(support_chauffe, ["imposed"])
    champ[0].add_to_component("imposed", valeur)
    return champ


# ANCHOR: donnees
# The left edge goes from 20 to 220 °C; the modulus halves on the way.
materiaux = pc.element_field.material_field(
    modele,
    [
        ("k", 1.0),
        ("rho", 1.0),
        ("cp", 1.0),
        ("E", 200e9),
        ("nu", 0.3),
        ("alpha", 1.2e-5),
        ("sigma_y", 250e6),
    ],
)
module = pc.Evolution(
    [(20.0, 200e9), (220.0, 100e9)],
    abscissa_type="T",
    ordinate_type="E",
    out_of_range="extrapolate",
)
data = {
    "model": modele,  # heat + mechanics + supports
    "times": [0.0, 100.0, 200.0, 400.0],  # the instants computed
    "materials": materiaux,  # a field, or an Evolution of fields
    "material_laws": module,  # E(T), applied to the temperature
    "loads": pc.Evolution(
        [(0.0, _temperature_imposee(20.0)), (100.0, _temperature_imposee(220.0))],
        out_of_range="clamp",
    ),
    "initial": {"temperature": pc.NodeField(maillage, ["T"]) + 20.0},
    "t_ref": 20.0,  # the stress-free temperature
}
# ANCHOR_END: donnees

# ANCHOR: lancement
pc.stepping.step_by_step(data)
for record in data["results"]:
    print(record["time"], record["iterations"], record["converged"])
# ANCHOR_END: lancement

assert [r["time"] for r in data["results"]] == [0.0, 100.0, 200.0, 400.0]
assert all(r["converged"] for r in data["results"])

# ANCHOR: resultats
dernier = data["results"][-1]
dernier["temperature"]  # T at the nodes
dernier["displacement"]  # u, multipliers left out
dernier["reactions"]  # the forces the supports exert, −Cᵀλ
dernier["behaviour"]  # stresses, internal variables, strains at the Gauss points
# ANCHOR_END: resultats

# Heated, the free plate expands without stress: the heat has spread to
# 220 °C everywhere by the last instant.
assert abs(dernier["temperature"].value(noeud(4, 2), "T") - 220.0) < 1e-3
assert (
    abs(dernier["displacement"].value(noeud(4, 2), "u_x") - 1.2e-5 * 200.0 * 2.0) < 1e-8
)
assert dernier["behaviour"].abs().max("sigma_xx") < 1e3

# ANCHOR: procedures
# The same computation, one procedure at a time: what step_by_step does.
options = pc.stepping.resolve_options(data)  # PAS_DEFA
probleme = pc.stepping.describe_problem(data, options)  # PAS_MODL
etat = pc.stepping.initial_state(data, probleme, options)  # PAS_INIT
instant, _ = pc.stepping.adjust_time_step(etat, 100.0, None)  # PAS_AJUSTE
chaleur = pc.stepping.heat_step(data, probleme, etat, instant, options)  # TRANSNON
meca = pc.stepping.mechanical_step(  # UNPAS
    data, probleme, etat, instant, chaleur.temperature, options
)
materiaux_fin = pc.stepping.materials_at(probleme, chaleur.temperature, instant)
# ANCHOR_END: procedures

assert meca.converged
assert abs(
    meca.solution.value(noeud(4, 2), "u_x")
    - data["results"][1]["displacement"].value(noeud(4, 2), "u_x")
) <= 1e-9 * abs(data["results"][1]["displacement"].value(noeud(4, 2), "u_x"))
assert materiaux_fin.min("E") < 200e9  # softened by the heat
resultats = {"results": []}
pc.stepping.store_step(resultats, probleme, etat, options, initial=True)  # PAS_RESU
assert resultats["results"][0]["time"] == 0.0
