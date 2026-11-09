# Ce que fait pyrucast

## En bref

pyrucast est une bibliothèque de **calcul par éléments finis**. Son cœur est
écrit en Rust, un langage compilé, rapide, qui répartit le travail sur tous
les cœurs du processeur ; on le pilote depuis **Python**. Un calcul est donc un
script, du premier point posé au dernier résultat tracé.

pyrucast n'a pas d'interface graphique : les figures sont produites par le
script lui-même, dans une fenêtre ou dans des fichiers d'image, et les
résultats s'exportent vers les logiciels de visualisation (ParaView, par le
format VTK). C'est un logiciel libre.

## Les étapes d'un calcul

Tout calcul suit les mêmes étapes, et chacune a son objet ou sa famille de
fonctions :

| étape | ce qu'on fait | dans pyrucast | vu dans |
|---|---|---|---|
| géométrie, maillage | poser des points, mailler lignes, surfaces et volumes | `Coords`, `pyrucast.mesh` | [Premiers pas](debutant.md), [Maillage](maillage.md) |
| espace éléments finis | choisir l'interpolation sur le maillage | `FiniteElementSpace` | [Premiers pas](debutant.md), [Régime stationnaire](thermique.md) |
| modèle | déclarer la physique, ses blocages, ses chargements | `pyrucast.model` | [Régime stationnaire](thermique.md) |
| matériaux | donner les coefficients que le modèle réclame | `pyrucast.element_field.material_field` | [Régime stationnaire](thermique.md) |
| second membre | intégrer les chargements | `pyrucast.node_field.external_forces` | [Régime stationnaire](thermique.md) |
| assemblage | construire la matrice du système | `pyrucast.matrix.stiffness` | [Régime stationnaire](thermique.md) |
| résolution | résoudre le système | `pyrucast.solver.solve` | [Régime stationnaire](thermique.md) |
| exploitation | tracer, extraire, exporter, sauvegarder | `plot`, `pyrucast.export.export_vtk`, `pyrucast.save` | [Maillage](maillage.md), [Régime stationnaire](thermique.md), [Thermique non linéaire](thermique-non-lineaire.md) |

Quand le problème n'est pas linéaire ou évolue dans le temps, le
[calcul pas à pas](../pas-a-pas.md) (`pyrucast.stepping.step_by_step`) répète
les étapes de l'assemblage à la résolution à chaque instant, jusqu'à
l'équilibre.

## Les domaines couverts

Chaque ligne renvoie à la page qui la détaille.

**Thermique.**
[Conduction](../thermique.md) isotrope, orthotrope ou anisotrope, en régime
stationnaire ou transitoire ; [convection et flux imposé](../echanges.md),
sources volumiques ; [rayonnement](../rayonnement.md) vers l'environnement.
Dans ce tutoriel : [régime stationnaire](thermique.md),
[régime transitoire](thermique-transitoire.md),
[rayonnement](thermique-non-lineaire.md).

**Diffusion.** La [loi de Fick](../diffusion.md), pour une ou plusieurs
espèces.

**Mécanique des milieux continus**, en contraintes planes, déformations planes,
axisymétrie ou 3D :

- [élasticité](../mecanique/elasticite.md) isotrope,
  [orthotrope ou anisotrope](../mecanique/orthotropie.md) ;
- plasticité de [von Mises](../mecanique/plasticite.md) (parfaite ou avec
  écrouissage isotrope), [Drucker-Prager et Ottosen](../mecanique/lois-plastiques.md),
  [Gurson](../mecanique/endommagement.md) ;
- [fluage et viscoplasticité](../mecanique/fluage.md) (Norton, Lemaitre,
  Blackburn, Chaboche) ;
- endommagement de [Mazars](../mecanique/mazars.md),
  [Damage TC et SiC/SiC](../mecanique/endommagement.md) ;
- dilatation thermique, [grandes transformations](../mecanique/grandes-transformations.md),
  dynamique.

**Éléments de structure.** [Barre](../mecanique/truss.md), poutres
d'[Euler-Bernoulli](../mecanique/bernoulli.md) et de
[Timoshenko](../mecanique/timoshenko.md) (portiques plans, cadres spatiaux),
[coques](../mecanique/coques.md).

**Conditions et liaisons.** [Blocages](../contraintes/dirichlet.md),
[relations entre nœuds](../contraintes/mpc.md),
[barres noyées dans un volume](../contraintes/embedded.md),
[contact](../contraintes/contact.md) sans frottement ; en égalité ou en
inégalité (voir [Contraintes](../contraintes.md)).

**Calcul pas à pas.** À chaque instant, la thermique puis la mécanique à la
température atteinte ; les lois non linéaires, le contact, la dynamique, le
pilotage du chargement — voir [Calcul pas à pas](../pas-a-pas.md).

**Éléments finis.** Quinze types, linéaires et quadratiques, du segment à
l'hexaèdre à 27 nœuds — voir [Éléments finis supportés](../elements/index.md).

**Maillage.** Des mailleurs intégrés — lignes et arcs, balayages,
triangulation de surfaces et de volumes, pavage en quadrangles, couche limite
d'hexaèdres — et la lecture des maillages gmsh et MED : voir
[Maillage](../operateurs/maillage.md) et [Mailler une géométrie](../mailler.md).

**Résultats.** [Tracés](../visualization.md) dans une fenêtre ou en PNG/SVG,
export VTK, [sauvegarde](../sauvegarde.md) d'une session dans un fichier
`.pyr`.

## Où trouver la documentation

- **Dans Python** : `help(pc.mesh.line)` affiche l'aide d'une fonction (voir
  [Python & conventions pyrucast](langage-python.md#appeler-une-fonction)).
  Les éditeurs de code la montrent aussi, et complètent les noms au fil de la
  frappe.
- **Ce livre** : les parties [Objets](../objets.md),
  [Détails des physiques](../physiques.md) et
  [Détail des opérateurs](../operateurs.md) décrivent chaque brique, avec sa
  théorie.
- **La référence de l'API Rust** :
  <https://pyrucast.github.io/pyrucast/rust/>.
- **Des exemples** complets, un par sujet : dossier
  [`examples/`](https://github.com/Pyrucast/pyrucast/tree/master/examples) du
  dépôt.

## Compiler pyrucast soi-même

La version précompilée suffit pour tout ce tutoriel. Compiler pyrucast depuis
ses sources ne sert qu'à l'installer avec la visualisation sur une plateforme
qui n'a pas de version précompilée, ou à travailler sur pyrucast lui-même. Il
faut alors Rust et l'outil `maturin` : la marche à suivre est dans
[Installation et démarrage rapide](../installation.md#compilation).

Suite : [Python & conventions pyrucast](langage-python.md).
