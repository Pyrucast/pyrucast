"""Formation débutant — 2. Calcul thermique. / Beginner training — 2. Thermal.

FR — Reprend la **chape percée** maillée au chapitre 1 : le script importe
directement `structured_mesh` de `formation/maillage.py`, donc le calcul est
mené sur le volume **HEX8 structuré**, sans redécrire la moindre cote.

EN — Picks the **pierced lug** meshed in chapter 1 back up: the script imports
`structured_mesh` straight from `formation/maillage.py`, so the analysis runs
on the **structured HEX8** volume without restating a single dimension.

FR — Équation résolue, conduction stationnaire : `div(-k·grad T) = q` dans le
volume, avec `T = T_imp` sur la partie bloquée du bord et
`-k·grad T · n = φ_imp + h·(T - T_ext)` sur le reste. Discrétisée, elle donne
le système `[K]{T} = {P}` que `solver.solve` factorise.

EN — Equation solved, steady conduction: `div(-k·grad T) = q` inside the
volume, with `T = T_imp` on the constrained part of the boundary and
`-k·grad T · n = φ_imp + h·(T - T_ext)` on the rest. Once discretised it
becomes the `[K]{T} = {P}` system that `solver.solve` factorises.

FR — Le calcul se mène en **deux temps**, chacun tracé avant d'être résolu :

1. **conduction seule** — température imposée sur l'alésage et flux imposé sur
   la face gauche ;
2. **conduction + convection + source** — on ajoute le film convectif sur le
   nez arrondi et la cartouche chauffante, sans rien retoucher au reste.

EN — The analysis runs in **two steps**, each plotted before being solved:

1. **conduction alone** — imposed temperature on the bore and imposed flux on
   the left face;
2. **conduction + convection + source** — the convective film on the rounded
   nose and the heating cartridge are added, nothing else changes.

FR — Les régions chargées sont repérées **par leur forme**, jamais par des
numéros de nœuds : la famille `pyrucast.mesher.points_*` sélectionne les nœuds
d'un plan, d'un cylindre, d'une sphère… et `pyrucast.mesher.elements_on`
remonte aux éléments qu'ils portent entièrement.

EN — The loaded regions are located **by shape**, never by node numbers: the
`pyrucast.mesher.points_*` family selects the nodes of a plane, a cylinder, a
sphere… and `pyrucast.mesher.elements_on` walks back to the elements they
fully carry.

FR — Attention, les contributions nodales de deux régions chargées adjacentes
ne se **somment pas** toutes seules : chaque chargement est assemblé sur son
propre support, et `|` juxtapose ces supports au lieu de les additionner — à
un nœud partagé, `solve` retient la valeur de la **première** zone qui définit
le couple (nœud, composante), et `|` ne lève une erreur que si les deux
valeurs diffèrent. `+` ne change rien : l'arithmétique de champs apparie elle
aussi les zones **par support**. Pour additionner vraiment deux régions qui se
touchent, il faut d'abord les ramener sur un support commun
(`pyrucast.field.restrict` sur un même maillage), puis `+`. Les quatre régions
utilisées ici sont deux à deux disjointes, ce qui évite la question.

EN — Beware, the nodal contributions of two adjacent loaded regions do **not**
add up on their own: each load is assembled on its own support, and `|`
juxtaposes those supports rather than summing them — at a shared node `solve`
keeps the value of the **first** zone defining the (node, component) pair, and
`|` only raises an error when the two values differ. `+` changes nothing:
field arithmetic likewise pairs zones **by support**. To genuinely add two
touching regions, bring them onto a common support first
(`pyrucast.field.restrict` onto one and the same mesh), then `+`. The four
regions used here are pairwise disjoint, which sidesteps the question.

FR — Limites connues : **pas de rayonnement** et **pas de terme transitoire**
— seule la conduction **stationnaire** est résolue ici. Voir la page thermique
du livre pour le détail de ces limites.

EN — Known limits: **no radiation** and **no transient term** — only **steady**
conduction is solved here. See the book's thermal page.

Lancement / Running ::

    maturin develop --release
    python formation/thermique.py

    # Figures du livre / book figures (book/src/formation/img/) :
    # PYRUCAST_FORMATION_IMG_DIR=book/src/formation/img python formation/thermique.py
"""

import os

import pyrucast as pc
from maillage import (
    HEIGHT,
    HOLE_RADIUS,
    LENGTH,
    OUT,
    THICKNESS,
    show,
    structured_mesh,
)

# FR — Vue commune aux figures du chapitre 1 : azimut, élévation, échelle.
# EN — The view shared by chapter 1's figures: azimuth, elevation, scale.
VUE = (-45, 25, 1.0)

# FR — Une couleur par région chargée, tenue d'une figure à l'autre.
# EN — One colour per loaded region, kept from one figure to the next.
BLEU = (0, 0, 255)  # alésage / bore
ROUGE = (255, 0, 0)  # face gauche / left face
TURQUOISE = (0, 190, 190)  # nez convecté / convected nose
VERT = (0, 170, 0)  # cartouche / cartridge

# ── Données physiques / Physical data ──────────────────────────────────────
K_COND = 50.0  # W/m/K
FLUX_IMPOSE = -40_000.0  # W/m², face gauche / left face
H_CONV, T_EXT = 240.0, -80.0  # W/m²/K, °C — convection, nez / nose
SOURCE_VOLUMIQUE = 4_000_000.0  # W/m³, cartouche / cartridge (≈ 200 W)
T_IMPOSEE = 250.0  # °C, alésage / bore

# FR — Cartouche chauffante : un cylindre fictif traversant l'épaisseur.
# EN — Heating cartridge: a fictitious cylinder through the thickness.
CARTOUCHE_X = 0.12  # m
RAYON_CARTOUCHE = 0.035  # m

# FR — L'axe du trou : la normale du plan de la pièce (Y), passant par le
# centre du demi-disque. Il sert deux fois — alésage et nez arrondi sont deux
# cylindres coaxiaux, de rayons différents.
# EN — The hole's axis: the part plane's normal (Y), through the half-disc
# centre. It serves twice — bore and rounded nose are two coaxial cylinders of
# different radii.
BAS_AXE = [LENGTH, -THICKNESS, HEIGHT / 2.0]
HAUT_AXE = [LENGTH, 2.0 * THICKNESS, HEIGHT / 2.0]


def show_nodefield(mesh: pc.Mesh, field: pc.NodeField, title: str, file: str) -> None:
    """FR — Trace `field` sur `mesh` : fenêtre interactive, ou SVG si le
    répertoire de figures est défini. Un seul appel par figure, donc un tracé
    du script = une image du livre.

    EN — Plot `field` over `mesh`: interactive window, or SVG when the figure
    directory is set. One call per figure, so one plot in the script = one
    image in the book.
    """
    mesh.plot(
        view=VUE,
        title=title,
        field=field,
        component="T",
        cmap="coolwarm",
        smooth=1,
        save=os.path.join(OUT, file) if OUT else None,
    )


# ── Étape 1 — conduction seule / Step 1 — conduction alone ─────────────────
# ANCHOR: regions_conduction
def regions_conduction(volume: pc.Mesh, peau: pc.Mesh) -> tuple[pc.Mesh, pc.Mesh]:
    """FR — Les deux régions du premier problème : l'alésage, tenu à une
    température, et la face gauche, qui reçoit un flux.

    EN — The two regions of the first problem: the bore, held at a given
    temperature, and the left face, which receives a flux.
    """
    # FR — L'alésage : les nœuds **sur** le cylindre de rayon `HOLE_RADIUS`.
    # `points_on_cylinder` exclut les disques d'extrémité (ce sont des faces
    # planes, `points_on_plane` est là pour ça) : on récupère exactement la
    # paroi du trou. Un blocage ne demandant que des **nœuds**, la sélection se
    # lit directement sur le volume — inutile d'en extraire la peau. Le
    # résultat étant déjà un maillage POI1, il sert tel quel de support ; seul
    # `consolidate` est nécessaire, pour écarter le sous-maillage vide que
    # laisse la partie du volume qui ne touche pas le trou.
    # EN — The bore: the nodes **on** the cylinder of radius `HOLE_RADIUS`.
    # `points_on_cylinder` leaves the end discs out (they are flat faces, and
    # `points_on_plane` is there for those): what comes back is exactly the
    # hole's wall. A constraint only needs **nodes**, so the selection reads
    # straight off the volume — no need to extract its boundary. The result
    # already being a POI1 mesh, it serves as the support as is; only
    # `consolidate` is needed, to drop the empty submesh left by the part of
    # the volume that does not touch the hole.
    alesage = pc.consolidate(
        pc.mesher.points_on_cylinder(volume, BAS_AXE, HAUT_AXE, HOLE_RADIUS)
    )

    # FR — La face gauche : les nœuds du plan x = 0, puis les QUA4 qu'ils
    # portent. Un flux s'intègre sur une **surface** : les nœuds ne suffisent
    # pas, `elements_on(..., strict=True)` remonte aux mailles dont **tous**
    # les nœuds sont sélectionnés. Le plan est infini, mais il ne coupe la peau
    # qu'à cet endroit.
    # EN — The left face: the nodes of the plane x = 0, then the QUA4 cells
    # they carry. A flux integrates over a **surface**: nodes are not enough,
    # `elements_on(..., strict=True)` walks back to the cells **all** of whose
    # nodes are selected. The plane is infinite, but it only meets the skin
    # there.
    noeuds_gauche = pc.mesher.points_on_plane(peau, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    face_gauche = pc.mesher.elements_on(peau, noeuds_gauche, strict=True)

    # FR — Une couleur par région, et la peau en fil de fer autour : la figure
    # se lit comme le schéma des conditions aux limites.
    # EN — One colour per region, the skin drawn as a wireframe around them:
    # the figure reads like the boundary-condition sketch.
    alesage.unit().face_color = BLEU
    face_gauche.unit().face_color = ROUGE
    show(
        peau | alesage | face_gauche,
        "Étape 1 — conditions aux limites",
        "thermique-cl-conduction.svg",
        wireframe=True,
    )
    return alesage, face_gauche


# ANCHOR_END: regions_conduction


# ANCHOR: conduction
def conduction(volume: pc.Mesh, alesage: pc.Mesh, face_gauche: pc.Mesh) -> pc.NodeField:
    """FR — Premier problème : conduction, température imposée, flux imposé.

    EN — First problem: conduction, imposed temperature, imposed flux.
    """
    fes = pc.FiniteElementSpace(volume)
    gauche_fes = pc.FiniteElementSpace(face_gauche)

    # FR — Le modèle de conduction porte les degrés de liberté « T » (primal)
    # et « q » (dual) sur tout le volume.
    # EN — The conduction model carries the "T" (primal) and "q" (dual) degrees
    # of freedom over the whole volume.
    modele = pc.Model.heat_conduction(fes)

    # FR — Le Dirichlet s'écrit sur deux maillages POI1 : le support bloqué
    # (l'alésage, tel que `points_on_cylinder` l'a renvoyé) et un jumeau dédié
    # aux multiplicateurs de Lagrange, obtenu par copie translatée de zéro —
    # deux jeux de nœuds distincts, donc deux jeux d'inconnues.
    # EN — Dirichlet is written on two POI1 meshes: the constrained support
    # (the bore, exactly as `points_on_cylinder` returned it) and a twin
    # dedicated to the Lagrange multipliers, obtained as a zero-translated
    # copy — two distinct node sets, hence two sets of unknowns.
    multiplicateur_T = pc.mesher.translate(alesage, [0.0, 0.0, 0.0])
    modele = modele | pc.Model.dirichlet("T", "q", alesage, multiplicateur_T)

    # FR — La conduction ne réclame qu'un coefficient, « k ».
    # EN — Conduction asks for a single coefficient, "k".
    materiaux = pc.build.material_field(modele, [("k", K_COND)])

    # FR — Flux imposé sur la face gauche.
    # EN — Imposed flux on the left face.
    flux_gauche = pc.assemble.flux(gauche_fes[0], FLUX_IMPOSE, "q")

    # FR — Température imposée sur l'alésage. La valeur se pose sur le maillage
    # des multiplicateurs, pas sur l'alésage lui-même.
    # EN — Imposed temperature on the bore. The value is set on the
    # multipliers' mesh, not on the bore itself.
    temperature_imposee = pc.NodeField(multiplicateur_T, ["imposed_T"])
    temperature_imposee[0].add_to_component("imposed_T", T_IMPOSEE)

    # FR — `[K]{T} = {P}` : la matrice assemblée, le second membre réuni par
    # `|`, et une factorisation LU creuse mise en cache.
    # EN — `[K]{T} = {P}`: the assembled matrix, the right-hand side gathered
    # by `|`, and a cached sparse LU factorisation.
    K = pc.assemble.stiffness(modele, materiaux)
    return pc.solver.solve(K, flux_gauche | temperature_imposee)


# ANCHOR_END: conduction


# ── Étape 2 — convection et source / Step 2 — convection and source ────────
# ANCHOR: regions_convection
def regions_convection(volume: pc.Mesh, peau: pc.Mesh) -> tuple[pc.Mesh, pc.Mesh]:
    """FR — Les deux régions ajoutées au second problème : la surface
    convectée et le volume chauffé.

    EN — The two regions added by the second problem: the convected surface
    and the heated volume.
    """
    # FR — Le nez arrondi : même axe que l'alésage, rayon du demi-disque. Une
    # convection s'intègre elle aussi sur une surface, d'où le `elements_on`.
    # EN — The rounded nose: same axis as the bore, half-disc radius.
    # Convection integrates over a surface too, hence the `elements_on`.
    noeuds_nez = pc.mesher.points_on_cylinder(peau, BAS_AXE, HAUT_AXE, HEIGHT / 2.0)
    nez = pc.mesher.elements_on(peau, noeuds_nez, strict=True)
    nez.unit().face_color = TURQUOISE
    show(
        peau | nez,
        "Étape 2 — surface convectée",
        "thermique-cl-convection.svg",
        wireframe=True,
    )

    # FR — La cartouche chauffante : cette fois les nœuds **dans** un cylindre
    # (`points_in_cylinder`, volume plein, disques d'extrémité compris), pris
    # sur le volume et non sur la peau, et les HEX8 qu'ils portent entièrement.
    # Le maillage étant structuré, la cartouche est approchée par un escalier
    # d'hexaèdres — c'est le prix du `strict=True`.
    # EN — The heating cartridge: this time the nodes **inside** a cylinder
    # (`points_in_cylinder`, solid volume, end discs included), taken on the
    # volume rather than on the skin, and the HEX8 cells they fully carry. The
    # mesh being structured, the cartridge comes out as a staircase of
    # hexahedra — that is what `strict=True` costs.
    noeuds_cartouche = pc.mesher.points_in_cylinder(
        volume,
        [CARTOUCHE_X, -THICKNESS, HEIGHT / 2.0],
        [CARTOUCHE_X, 2.0 * THICKNESS, HEIGHT / 2.0],
        RAYON_CARTOUCHE,
    )
    cartouche = pc.consolidate(
        pc.mesher.elements_on(volume, noeuds_cartouche, strict=True)
    )
    cartouche.unit().face_color = VERT
    show(
        peau | cartouche,
        "Étape 2 — cartouche chauffante",
        "thermique-cl-source.svg",
        wireframe=True,
    )
    return nez, cartouche


# ANCHOR_END: regions_convection


# ANCHOR: complet
def conduction_convection_source(
    volume: pc.Mesh,
    alesage: pc.Mesh,
    face_gauche: pc.Mesh,
    nez: pc.Mesh,
    cartouche: pc.Mesh,
) -> pc.NodeField:
    """FR — Second problème : le premier, plus la convection et la source.

    EN — Second problem: the first one, plus convection and source.
    """
    fes = pc.FiniteElementSpace(volume)
    nez_fes = pc.FiniteElementSpace(nez)
    gauche_fes = pc.FiniteElementSpace(face_gauche)
    cartouche_fes = pc.FiniteElementSpace(cartouche)

    # FR — La convection n'est pas un système à part : son terme `h·Nᵢ·Nⱼ`
    # s'ajoute **dans** la matrice de conduction. Les sous-modèles se
    # réunissent donc par `|`, sur les mêmes degrés de liberté « T » et « q ».
    # EN — Convection is not a separate system: its `h·Nᵢ·Nⱼ` term adds **into**
    # the conduction matrix. The sub-models therefore join with `|`, on the
    # same "T" and "q" degrees of freedom.
    modele = pc.Model.heat_conduction(fes) | pc.Model.convection(nez_fes)
    multiplicateur_T = pc.mesher.translate(alesage, [0.0, 0.0, 0.0])
    modele = modele | pc.Model.dirichlet("T", "q", alesage, multiplicateur_T)

    # FR — Un seul champ matériau pour tout le modèle : « k » est demandé par
    # la conduction, « h » par la convection.
    # EN — A single material field for the whole model: "k" is required by
    # conduction, "h" by convection.
    materiaux = pc.build.material_field(modele, [("k", K_COND), ("h", H_CONV)])

    # FR — Les deux chargements de l'étape 1, inchangés.
    # EN — Step 1's two loads, unchanged.
    flux_gauche = pc.assemble.flux(gauche_fes[0], FLUX_IMPOSE, "q")
    temperature_imposee = pc.NodeField(multiplicateur_T, ["imposed_T"])
    temperature_imposee[0].add_to_component("imposed_T", T_IMPOSEE)

    # FR — Terme externe de la convection, h·T_ext : même opérateur `flux`, sur
    # la surface convectée. La part en T, elle, est déjà dans la matrice.
    # EN — The convection's external term, h·T_ext: the same `flux` operator,
    # on the convected surface. The T-dependent part is already in the matrix.
    charge_convection = pc.assemble.flux(nez_fes[0], H_CONV * T_EXT, "q")

    # FR — Source volumique dans la cartouche. `flux` est l'unique opérateur
    # de charge répartie de pyrucast : la dimension de l'intégrale est celle
    # des éléments qu'on lui donne, ici des HEX8, donc une densité volumique.
    # EN — Volume source inside the cartridge. `flux` is pyrucast's only
    # distributed-load operator: the integral's dimension is that of the
    # elements handed in, HEX8 here, hence a volume density.
    charge_source = pc.assemble.flux(cartouche_fes[0], SOURCE_VOLUMIQUE, "q")

    # FR — Union : les quatre régions chargées étant disjointes, chaque couple
    # (nœud, composante) n'est défini que par une zone — `|` les juxtapose sans
    # rien avoir à arbitrer. Sur des régions qui se touchent, il faudrait
    # d'abord les ramener sur un support commun (`field.restrict`) et les
    # sommer par `+` : `|` ne somme pas les contributions d'un nœud partagé.
    # EN — Union: the four loaded regions being disjoint, every (node,
    # component) pair is defined by a single zone — `|` juxtaposes them with
    # nothing to arbitrate. For touching regions one would first bring them
    # onto a common support (`field.restrict`) and sum with `+`: `|` does not
    # add up the contributions at a shared node.
    second_membre = (
        flux_gauche | charge_convection | charge_source | temperature_imposee
    )

    K = pc.assemble.stiffness(modele, materiaux)
    return pc.solver.solve(K, second_membre)


# ANCHOR_END: complet


def main() -> None:
    # FR — Le maillage du chapitre 1, tel quel : aucune cote n'est redonnée
    # ici, le volume vient de `formation/maillage.py`. `plot=False` : on veut
    # le maillage, pas les figures du chapitre 1.
    # EN — Chapter 1's mesh, as is: not one dimension is restated here, the
    # volume comes from `formation/maillage.py`. `plot=False`: we want the
    # mesh, not chapter 1's figures.
    _, volume = structured_mesh(plot=False)

    # FR — Les chargements **répartis** s'intègrent sur des faces et non sur
    # des nœuds : il leur faut de vraies mailles de bord, que `skin` extrait du
    # volume. `consolidate` ramène la peau à un seul sous-maillage, pour que
    # les sélections qui suivent en renvoient un seul elles aussi.
    # EN — **Distributed** loads integrate over faces and not over nodes: they
    # need genuine boundary cells, which `skin` extracts from the volume.
    # `consolidate` brings the boundary back to a single submesh, so that the
    # selections below return a single one as well.
    peau = pc.consolidate(pc.mesher.skin(volume))

    alesage, face_gauche = regions_conduction(volume, peau)
    t_conduction = conduction(volume, alesage, face_gauche)
    show_nodefield(
        volume, t_conduction, "Étape 1 — température (°C)", "thermique-conduction.svg"
    )

    nez, cartouche = regions_convection(volume, peau)
    t_complet = conduction_convection_source(
        volume, alesage, face_gauche, nez, cartouche
    )
    show_nodefield(
        volume, t_complet, "Étape 2 — température (°C)", "thermique-complet.svg"
    )

    print(f"volume         : {volume.cell_count()} HEX8")
    print(f"face gauche    : {face_gauche.cell_count()} QUA4")
    print(f"nez convecté   : {nez.cell_count()} QUA4")
    print(f"cartouche      : {cartouche.cell_count()} HEX8")
    print(
        f"étape 1 : T min = {t_conduction.min('T'):.1f} °C, "
        f"T max = {t_conduction.max('T'):.1f} °C"
    )
    print(
        f"étape 2 : T min = {t_complet.min('T'):.1f} °C, "
        f"T max = {t_complet.max('T'):.1f} °C"
    )


if __name__ == "__main__":
    main()
