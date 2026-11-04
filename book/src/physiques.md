# Détails des physiques

Une **physique** est un sous-modèle (`SubModel`) que construit un opérateur de
`pyrucast.model`. L'opérateur reçoit son support, un
[espace éléments finis](fe-space.md) ou les maillages que relie une contrainte,
et rend un [modèle](model.md) qui le couvre en entier : un sous-modèle par
zone. Les physiques se composent par `|` : une conduction et la contrainte qui
fige sa température sur un bord forment un seul modèle.

Cette page dit ce qu'ont en commun toutes les physiques : leur **tag**, leur
**type**, leurs **variables**, et ce que chacune sait produire. Chaque page
qui suit décrit une physique selon le même plan (voir
[le plan des pages](#le-plan-des-pages)).

## Les tags

Chaque sous-modèle porte un ou plusieurs **tags**, qui disent à quelle
physique il appartient. `sub.physics()` les donne, et un modèle se
restreint à un tag (voir [Modèle](model.md#nature-physique-et-filtrage)).
Les pages sont rangées par tag :

| tag | physiques |
|---|---|
| `thermal` | conduction ; rayonnement, qui porte aussi `radiation` |
| `diffusion` | diffusion de Fick |
| `mechanical` | élasticité, plasticité, endommagement, barre, poutres, coques |
| `constraint` | Dirichlet, multi-points, baignage, contact |

Les échanges et le flux imposé n'ont pas de tag propre : ils prennent celui de
la physique qu'ils visent, `thermal` sur une conduction, `mechanical` sur une
élasticité. Une contrainte porte `constraint` **et** le tag de la physique dont
elle fige les inconnues.

## Les types de modèle

Une physique est de l'un de trois types, selon ce qu'elle sait faire :

| type | ce qu'il fait | physiques |
|---|---|---|
| **domaine avec comportement** | intègre sur un sous-espace éléments finis, lit un matériau, et porte une **loi** : de ce que la solution produit (gradient, déformation, courbures), elle calcule un flux, une contrainte ou des efforts, plus un éventuel état | conduction, rayonnement, Fick, élasticité, plasticité, endommagement, barre, poutres, coques |
| **domaine sans comportement** | intègre sur un sous-espace et lit un matériau, sans loi | échange de bord, transfert d'interface, flux imposé |
| **contrainte** | impose des relations linéaires entre inconnues par des multiplicateurs de Lagrange ; aucun sous-espace, aucun matériau | Dirichlet, multi-points, baignage, contact |

Ce que les accesseurs d'un sous-modèle répondent :

| | domaine avec comportement | domaine sans comportement | contrainte |
|---|---|---|---|
| `label()` | le nom de la physique : `"HeatConduction"`, `"Elasticity"`… | `"BoundaryTransfer"`, `"InterfaceTransfer"`, `"Flux"` | `"Dirichlet"`, `"Mpc"`, `"Embedded"`, `"Contact"` |
| `material_components()` | les composantes matériau requises | les composantes requises | `None` |
| `fespace()` | le sous-espace où la loi est intégrée | `None` | `None` |
| `has_behavior` | `True` | `True` | `False` |
| `multiplier_mesh()` | vide | vide | les nœuds multiplicateurs, un par relation |

`has_behavior` répond donc `True` pour tout domaine, et `fespace()` ne rend
le sous-espace que d'un domaine avec comportement.

Seul un domaine avec comportement se prête aux opérateurs de comportement de
`pyrucast.element_field` : `strain`, `integrate_behavior`, et la tangente
`pyrucast.matrix.tangent` (voir [Comportement](operateurs/comportement.md)).

## Les variables

Chaque physique nomme ses variables :

- les **primales** (`primal_vars()`) sont les inconnues, les **colonnes** de
  la matrice : `T`, `u_x`… ;
- les **duales** (`dual_vars()`) sont les **lignes**, où se lit le second
  membre : `q`, `f_x`… La duale d'une primale est ce qui travaille avec elle :
  un flux de chaleur avec une température, une force avec un déplacement ;
- le **matériau** est un [champ aux points de Gauss](element-field.md) qui
  porte les composantes **requises** (`material_components()`) et, au
  besoin, des composantes **facultatives** ;
- l'**entrée du comportement** est ce que `strain(modele, solution, materiaux)`
  calcule depuis la solution : un gradient, une déformation, des courbures ;
- la **sortie du comportement** est ce que rend
  `integrate_behavior(modele, entrée, materiaux, prev, dt)` : le flux, la
  contrainte ou les efforts, puis l'**état** que la loi relira au pas suivant
  (`prev`) ;
- les **forces internes** (`pyrucast.node_field.internal_forces`) sont ce
  que la sortie exerce sur les duales : `∫ Bᵀσ`, où `B` est l'opérateur qui
  produit l'entrée depuis la solution. Chaque page dit comment sa physique les
  calcule.

Une contrainte n'a qu'une primale, `lambda` (le multiplicateur), et une
duale, `imposed` (la ligne de la relation, où s'écrit la valeur imposée).

## Ce que chaque physique produit

Les tableaux disent, pour chaque physique, les matrices qu'elle apporte et
comment s'obtient sa tangente. Un tiret dit qu'elle n'y contribue rien : une
masse assemblée sur un modèle qui contient un échange ignore simplement
l'échange.

La **tangente** est ce que rend `pyrucast.matrix.tangent`. « = raideur » dit
que la physique répond à une demande de tangente par sa raideur ;
« analytique » qu'elle a une formule ; « perturbation » qu'elle la calcule par
différences finies sur sa loi ; « aucune » qu'elle refuse la demande.

### `thermal`

| physique | type | raideur | masse | géométrique | tangente |
|---|---|---|---|---|---|
| [`heat_conduction`](thermique.md) | avec comportement | conduction | capacité `ρ·cp` | — | = raideur |
| [`radiation`](rayonnement.md) | avec comportement | film linéarisé | — | — | analytique |

### `diffusion`

| physique | type | raideur | masse | géométrique | tangente |
|---|---|---|---|---|---|
| [`fick`](diffusion.md) | avec comportement | diffusion | stockage `poro` | — | = raideur |

### Le tag de la cible

| physique | type | raideur | masse | géométrique | tangente |
|---|---|---|---|---|---|
| [`boundary_transfer`](echanges.md#échange-de-bord) | sans comportement | film | — | — | = raideur |
| [`interface_transfer`](echanges.md#transfert-dinterface) | sans comportement | quatre blocs | — | — | = raideur |
| [`flux`](echanges.md#flux-imposé) | sans comportement | — | — | — | — |

### `mechanical`

| physique | type | raideur | masse | géométrique | tangente |
|---|---|---|---|---|---|
| [`elasticity`](mecanique/elasticite.md) | avec comportement | oui | oui | oui | analytique |
| [`plasticity_perfect`, `plasticity_isotropic`](mecanique/plasticite.md) | avec comportement | élastique | oui | oui | analytique |
| [`drucker_prager`, `ottosen`](mecanique/lois-plastiques.md), [`gurson`](mecanique/endommagement.md) | avec comportement | élastique | oui | oui | perturbation |
| [`creep_norton`, `creep_blackburn`, `creep_lemaitre`, `viscoplasticity_chaboche`, `viscoplasticity_lemaitre_chaboche`](mecanique/fluage.md) | avec comportement | élastique | oui | oui | perturbation |
| [`mazars`](mecanique/mazars.md), [`damage_tc`, `damage_sic_sic`](mecanique/endommagement.md) | avec comportement | non endommagée | oui | oui | aucune |
| [`truss`](mecanique/truss.md) | avec comportement | oui | oui | oui | = raideur |
| [`bernoulli`](mecanique/bernoulli.md), [`timoshenko`](mecanique/timoshenko.md) | avec comportement | oui | oui | oui, sauf en 1-D | = raideur |
| [`shell`](mecanique/coques.md) | avec comportement | oui | — | — | = raideur |

### `constraint`

| physique | type | raideur | masse | géométrique | tangente |
|---|---|---|---|---|---|
| [`dirichlet`](contraintes/dirichlet.md), [`mpc`](contraintes/mpc.md), [`embedded`](contraintes/embedded.md), [`contact`](contraintes/contact.md) | contrainte | blocs `C` et `Cᵀ` | — | — | = raideur |

## Le plan des pages

Chaque physique est décrite dans cet ordre :

1. **En bref** : ce qu'elle modélise, son opérateur Python, son tag, son
   type, et les éléments qu'elle accepte ;
2. **Théorie** : les équations, de la forme locale à la matrice élémentaire ;
3. **Variables** : primales, duales, matériau, entrée et sortie du
   comportement, état ;
4. **Ce qu'elle produit** : matrices, tangente, forces internes et externes ;
5. **Exemple** en Python ;
6. **Côté Rust**.

Une page qui regroupe plusieurs lois d'une même physique (les lois
plastiques, de fluage, d'endommagement) donne l'en-bref et les variables
communes une fois, puis la théorie et le matériau loi par loi.

## Côté Rust

Une physique est un `struct` de `src/models/` qui implémente le trait
`SubModelKind` : ses variables (`primal_vars`, `dual_vars`), ses tags
(`physics`), son nom (`label`), et ses capacités, qu'elle expose par trois
méthodes :

- `as_domain()` rend `Some(&dyn Domain)` pour un domaine : le sous-espace où
  lire le matériau (`material_fespace`), les composantes requises et
  facultatives, les matrices élémentaires ;
- `as_behavior()` rend `Some(&dyn Behavior)` pour un domaine qui porte une
  loi : son entrée (`deformation_reads`, produite par `strain_operator`), sa
  sortie (`behavior_output_components`), l'état qu'elle relit (`state_reads`),
  sa tangente au point (`tangent_point`, `tangent_source`), et `requires_dt`
  pour une loi qui dépend du pas de temps ;
- `as_constraint()` rend `Some(&dyn Constraint)` pour une contrainte : ses
  nœuds multiplicateurs et ses relations.

Le code générique (assembleur, comportement, sauvegarde) ne passe que par ce
trait : `SubModel::as_kind` est le seul `match` sur les physiques. Les
matrices qu'une physique apporte se déclarent par `matrix_layout(kind)`, une
par genre (`MatrixKind` : raideur, masse, géométrique, tangente) ; une
contrainte ou un transfert d'interface rend ses blocs par `contributions`.
Pour en écrire une, voir [Ajouter une physique](ajouter-une-physique.md).
