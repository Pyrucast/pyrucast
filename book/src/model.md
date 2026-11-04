# Modèle physique (Model/SubModel)

Un modèle décrit **le problème à résoudre** : quelles physiques, sur quelles
zones du maillage, et quelles contraintes les tiennent. Il ne porte aucune
valeur de matériau : la conductivité, le module d'Young ou la densité d'un
chargement arrivent dans un [champ matériau](element-field.md) au moment de
l'assemblage, et un même modèle sert avec d'autres valeurs. C'est un
[agrégat](aggregate.md) :

- un **sous-modèle** (`SubModel`) est une physique sur une zone, c'est-à-dire
  sur un sous-espace d'un [espace éléments finis](fe-space.md), ou bien une
  contrainte ;
- un **modèle** (`Model`) est une liste de sous-modèles, de physiques
  éventuellement différentes : une conduction et une élasticité sur le même
  maillage, avec leurs appuis.

Le modèle est la troisième couche, posée sur les deux autres :

| Couche | Objets | Ce qu'elle décrit |
|---|---|---|
| Géométrie | [`Mesh` / `SubMesh`](mesh.md) | les nœuds et les mailles |
| Interpolation | [`FiniteElementSpace` / `SubFiniteElementSpace`](fe-space.md) | les fonctions de forme et les points de Gauss |
| Physique | `Model` / `SubModel` | les équations, leurs inconnues, leurs contraintes |

## Déclarer un modèle

Chaque physique se déclare par une fonction de `pyrucast.model`, qui reçoit
l'espace éléments finis et rend un modèle couvrant **tout** l'espace : un
sous-modèle par sous-espace. Une contrainte reçoit, au lieu d'un espace, les
maillages qu'elle relie, et rend un seul sous-modèle, quel que soit leur nombre
de zones.

On compose les modèles avec `|`, qui rend un nouveau modèle : les sous-modèles
des deux côtés, dans l'ordre, sans répéter un sous-modèle déjà présent.
`Model()` est un modèle vide ; `add_sub` et `add_subs` ajoutent des
sous-modèles en place. Tous les sous-modèles d'un modèle vivent dans un même
`Coords` : en composer deux de `Coords` différents est refusé.

Certains termes agissent sur les inconnues d'une autre physique : les
contraintes (`dirichlet`, `mpc`, `embedded`, `contact`), les échanges
(`boundary_transfer`, `interface_transfer`), le rayonnement (`radiation`) et le
chargement réparti (`flux`). Ils reçoivent cette physique, leur **cible** :
elle donne la ligne où écrire leur effet (`q` pour une contrainte sur `T`,
`f_x` pour une contrainte sur `u_x`), et garantit qu'une physique du problème
assemble bien cette ligne.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:model_api}}
```

`modele[i]` rend une **vue** sur un sous-modèle. Une vue se lit (variables,
natures, sous-espace, matériau attendu) mais ne se modifie pas : on change un
modèle en le recomposant.

| `Model` | |
|---|---|
| `len(modele)`, `modele[i]`, `modele[i:j]`, `for sub in modele`, `unit()`, `\|`, `add_sub`, `add_subs` | l'interface des [agrégats](aggregate.md) |
| `primal_vars()`, `dual_vars()` | les inconnues et les duales de tous les sous-modèles, dans l'ordre de première apparition |
| `dual_of(variable)` | la duale associée à une inconnue, ou `None` |
| `fespace()` | l'espace sur lequel le modèle intègre : le sous-espace de chaque physique, dans l'ordre ; deux physiques sur un même sous-espace le citent deux fois, une contrainte n'en a pas |
| `filter(nature)`, `exclude(nature)` | les sous-modèles qui portent, ou ne portent pas, cette nature |
| `multiplier_mesh()` | les nœuds-multiplicateurs de toutes les contraintes |
| `constraint_rhs([(nœud, g), …])`, `constraint_rhs_by_index([(i, g), …])` | les valeurs imposées, sur un modèle qui porte une seule contrainte (voir [Assembler et résoudre](#assembler-et-résoudre)) |
| `stiffness_matrix`, `mass_matrix`, `geometric_matrix`, `tangent_matrix`, `material_field`, `material_field_per_sub_model`, `strain`, `integrate_behavior`, `elastic_stress`, `elastic_strain`, `internal_forces`, `external_forces`, `update_contact` | les opérateurs qui prennent un modèle en premier argument, en méthode : `modele.stiffness_matrix(materiaux)` est `matrix.stiffness(modele, materiaux)` |
| `dump()` | le détail de chaque sous-modèle |

| `SubModel` (vue) | |
|---|---|
| `label()` | le nom de la physique : `"Elasticity"`, `"Dirichlet"`… |
| `physics()` | ses natures |
| `primal_vars()`, `dual_vars()` | ses inconnues et ses duales |
| `fespace()` | son sous-espace, ou `None` pour une contrainte |
| `material_components()` | les composantes matériau qu'il demande, ou `None` |
| `has_behavior` | vrai s'il porte une loi de comportement, que `integrate_behavior` intègre |
| `multiplier_mesh()` | ses nœuds-multiplicateurs, vide hors contrainte |
| `material_field([(c, v), …])` | un champ matériau uniforme pour ce seul sous-modèle |

## Les physiques disponibles

Le tableau donne, pour chaque déclaration, les variables qu'elle apporte au
modèle. Les équations, le matériau et les exemples sont dans la page de chaque
physique ; ce que chacune sait assembler est résumé dans
[Détails des physiques](physiques.md#ce-que-chaque-physique-produit).

| Déclaration (`pyrucast.model.…`) | Inconnues | Duales | Page |
|---|---|---|---|
| `heat_conduction(fes, symétrie)` | `T` | `q` | [Thermique](thermique.md) |
| `radiation(fes, cible)` | `T` | `q` | [Rayonnement](rayonnement.md) |
| `fick(fes, espèce, symétrie)` | `c_<espèce>` | `j_<espèce>` | [Diffusion](diffusion.md) |
| `boundary_transfer(fes, cible, variables)` | les `variables` | celles de la cible | [Échanges](echanges.md) |
| `interface_transfer(côté_a, côté_b, cible, variables, tol)` | les `variables` | celles de la cible | [Échanges](echanges.md) |
| `flux(fes, cible, duale)` | aucune | `duale` ; `"n"`, une pression, écrit dans `f_x`, `f_y`(, `f_z`) | [Flux imposé](echanges.md#flux-imposé) |
| `elasticity(fes, cinématique, symétrie, mesure)` | `u_x, u_y`(, `u_z`) | `f_x, f_y`(, `f_z`) | [Élasticité](mecanique/elasticite.md), [orthotropie](mecanique/orthotropie.md) |
| `plasticity_perfect`, `plasticity_isotropic`, `drucker_prager`, `ottosen`, `gurson`, `creep_norton`, `creep_blackburn`, `creep_lemaitre`, `viscoplasticity_chaboche`, `viscoplasticity_lemaitre_chaboche` `(fes, cinématique, mesure)` | `u_x, u_y`(, `u_z`) | `f_x, f_y`(, `f_z`) | [Plasticité](mecanique/plasticite.md), [lois d'écoulement](mecanique/lois-plastiques.md), [fluage](mecanique/fluage.md) |
| `mazars`, `damage_tc`, `damage_sic_sic` `(fes, cinématique, mesure)` | `u_x, u_y`(, `u_z`) | `f_x, f_y`(, `f_z`) | [Mazars](mecanique/mazars.md), [lois d'endommagement](mecanique/endommagement.md) |
| `truss(fes, mesure)` | `u_x, u_y`(, `u_z`) | `f_x, f_y`(, `f_z`) | [Barre](mecanique/truss.md) |
| `bernoulli(fes)`, `timoshenko(fes)` | 1-D : `w, theta` ; plan : `u_x, u_y, r_z` ; espace : `u_x…u_z, r_x…r_z` | `f_w, m_theta` ; `f_x, f_y, m_z` ; `f_x…f_z, m_x…m_z` | [Euler-Bernoulli](mecanique/bernoulli.md), [Timoshenko](mecanique/timoshenko.md) |
| `shell(fes, formulation)` | `u_x…u_z, r_x…r_z` | `f_x…f_z, m_x…m_z` | [Coques](mecanique/coques.md) |
| `dirichlet(maillage, cible, variable, sens)` | `lambda` | `imposed` | [Dirichlet](contraintes/dirichlet.md) |
| `mpc(termes, cible, sens)` | `lambda` | `imposed` | [Multi-points](contraintes/mpc.md) |
| `embedded(immergé, hôte, cible, variables, tol)` | `lambda` | `imposed` | [Baignage](contraintes/embedded.md) |
| `contact(esclave, maître, cible, variables)` | `lambda` | `imposed` | [Contact](contraintes/contact.md) |

`symétrie` (`"isotropic"`), `mesure` (`"linear"`), `sens` (`"="`) et `tol` ont
une valeur par défaut. Une poutre lit sa configuration (1-D, plan, espace) sur
le maillage. Les composantes matériau dépendent de la physique et de ses
options : `sub.material_components()` les donne.

`interface_transfer` rend un sous-modèle par paire de sous-espaces des deux
côtés, pris dans l'ordre. `update_contact(modele, déplacement)` ne déclare
rien : il rend le modèle avec ses contacts appariés de nouveau sur la
configuration déplacée (voir [Contact](contraintes/contact.md)).

## Inconnues et multiplicateurs

Chaque physique déclare :

- ses **inconnues**, ou variables primales (`primal_vars()`) : les composantes
  de la solution, et les colonnes de la matrice ;
- ses **duales** (`dual_vars()`) : les composantes du second membre, et les
  lignes de la matrice. `T` a pour duale `q`, `u_x` a pour duale `f_x`.

Une inconnue de la matrice est un couple (nœud, nom). Deux physiques qui
emploient le même nom au même nœud partagent l'inconnue : c'est ainsi qu'un
échange de bord sur `T` s'ajoute à la conduction, et que deux zones
d'élasticité se raccordent sur leurs nœuds communs. Deux noms différents ne se
rencontrent jamais : une conduction et une élasticité sur le même maillage
forment, dans la même matrice, deux problèmes que rien ne lie.

Une **contrainte** impose ses relations par multiplicateurs de Lagrange (voir
[Contraintes](contraintes.md)). À sa construction, elle crée dans le même
`Coords` un nœud par relation, le **nœud-multiplicateur**, à la position d'un
nœud de la relation : le nœud imposé d'un Dirichlet, le nœud esclave d'un
contact. Ce nœud porte l'inconnue `lambda`, dont la valeur à la solution est
la réaction, et la duale `imposed`, la ligne où s'écrit la valeur imposée.
Toutes les contraintes emploient ces deux noms sans se mélanger, puisque
chacune a ses propres nœuds. `multiplier_mesh()` rend ces nœuds, en zones
POI1.

## Assembler et résoudre

`matrix.stiffness(modele, materiaux)` rend **une seule** matrice pour tout le
modèle : physiques et contraintes, inconnues et multiplicateurs. Les
conditions aux limites sont des sous-modèles comme les autres. Le second
membre réunit trois sources :

- **les données que porte le modèle**, rendues par
  `node_field.external_forces(modele, materiaux)` : un chargement réparti
  (`flux`, dont la densité `phi_<duale>` est lue dans le matériau), la partie
  ambiante d'un échange de bord, le jeu d'un contact. Un modèle qui n'en a pas
  rend un champ vide ;
- **les valeurs imposées par une contrainte**, rendues par
  `contrainte.constraint_rhs([(nœud, g), …])` : on désigne chaque relation par
  un de ses nœuds, et la valeur `g` est écrite à son nœud-multiplicateur, dans
  la composante `imposed`. Le modèle doit porter une seule contrainte ;
  `constraint_rhs_by_index` désigne la relation par son rang (voir
  [Contraintes](contraintes.md#second-membre--le-helper-constraint_rhs)) ;
- **ce qu'on écrit soi-même** : un [champ aux nœuds](node-field.md) sur les
  composantes de `dual_vars()`, comme une force ou un flux ponctuels.

Deux champs posés sur le même maillage s'additionnent avec `+`, composante par
composante. `|` réunit des champs qui ne se recouvrent pas, comme les valeurs
imposées, portées par les nœuds-multiplicateurs.

```python
{{#include ../../tests/python/test_doc_conteneurs.py:model_chargement}}
```

Les opérateurs sont détaillés dans [Assemblage](operateurs/assemblage.md) et
[Solveur](operateurs/solveur.md). Des exemples complets à solution analytique
sont déroulés dans [Dirichlet](contraintes/dirichlet.md) (Poisson 1-D),
[Conduction thermique](thermique.md) et [Mécanique](mecanique.md).

Une physique qui n'a pas un genre de matrice n'y contribue pas :
`matrix.mass` sur une conduction avec échange de bord ne prend que la capacité
de la conduction.

## Nature physique et filtrage

Chaque sous-modèle déclare un **ensemble de natures** (`physics()`), qui dit à
quelle physique il appartient :

| Nature | Sous-modèles |
|---|---|
| `"mechanical"` | `elasticity`, les lois de plasticité et d'endommagement, `truss`, `bernoulli`, `timoshenko`, `shell` |
| `"thermal"` | `heat_conduction` ; `radiation` |
| `"radiation"` | `radiation`, en plus de `"thermal"` |
| `"diffusion"` | `fick` |
| `"constraint"` | `dirichlet`, `mpc`, `embedded`, `contact` |
| `"other"` | aucune physique ; sert, en Rust, à classer un bloc de matrice monté à la main |

Un `flux` et les échanges prennent la nature de leur cible : sur une
conduction, ils sont `"thermal"`, sur une diffusion, `"diffusion"`. Une
contrainte porte `"constraint"` **et** la nature de la physique dont elle fige
les inconnues : un Dirichlet sur `u_x` est `["constraint", "mechanical"]`. Une
relation qui lie les inconnues de deux physiques différentes ne porte que
`"constraint"` : aucun problème d'une seule physique ne saurait la résoudre.

`filter(nature)` garde les sous-modèles qui portent cette nature, et
`exclude(nature)` ceux qui ne la portent pas. Les deux se composent, et les
sous-modèles sont partagés, pas copiés. Comme une contrainte porte la nature
de ce qu'elle fige, `filter("mechanical")` rend la mécanique **avec** ses
appuis, un problème qui se résout tel quel, et
`filter("mechanical").exclude("constraint")` la mécanique **seule** :

```python
{{#include ../../tests/python/test_doc_conteneurs.py:model_natures}}
```

Les natures disent ce qu'un sous-modèle fige, pas ce qu'il est : un contact et
un Dirichlet sur les déplacements portent tous deux
`["constraint", "mechanical"]`. `label()` nomme la physique (`"Dirichlet"`,
`"Contact"`…) ; c'est par lui que le [pas à pas](pas-a-pas.md) reconnaît le
contact.

Chaque bloc d'une matrice assemblée garde les natures du sous-modèle qui l'a
produit : une matrice se sélectionne de la même façon (voir
[Matrice](matrix.md)).

## Ce qui est vérifié, et quand

À la **déclaration** :

- l'espace convient à la physique : une poutre d'Euler-Bernoulli demande un
  espace `"HERMITE3"`, une poutre de Timoshenko un espace `"MODEL_EMBEDDED"` ;
- la variable qu'une contrainte fige est une inconnue de sa cible ;
- la composition refuse un sous-modèle d'un autre `Coords`.

À l'**assemblage** : le champ matériau porte, sur le sous-espace de chaque
sous-modèle, les composantes qu'il demande. Le matériau n'est connu qu'à ce
moment-là.

## Ce qui est mis en cache

Un modèle garde, pour chaque genre de matrice (raideur, masse, raideur
géométrique, tangente), le **motif** de sa matrice : quelles lignes et quelles
colonnes portent un terme, et à quelle place de la matrice creuse chaque terme
élémentaire vient s'ajouter. Il est calculé au premier assemblage de ce genre,
puis repris tel quel : les assemblages suivants ne calculent que les valeurs.

Ce motif ne dépend que des inconnues que relient les sous-modèles, pas du
matériau, de la position des nœuds ou de l'état mécanique. Un calcul non
linéaire, qui réassemble sa tangente à chaque itération, le réutilise donc à
chaque fois. Le calculer représente plus de la moitié du premier assemblage :
sur un million de quadrangles en élasticité plane (4 cœurs), le premier
assemblage prend 1,9 s, les suivants 0,7 s.

Le motif est oublié quand on ajoute un sous-modèle (`add_sub`, `add_subs`). Un
nouveau modèle, rendu par `|`, `filter`, `exclude`, une tranche
`modele[i:j]` ou `update_contact`, part sans motif : dans une boucle, on garde
le modèle dans une variable plutôt que de le recomposer à chaque tour.

## Sauvegarde

Un modèle se [sauvegarde](sauvegarde.md) avec tout ce qu'il désigne : ses
espaces, leurs maillages, le `Coords`, les nœuds-multiplicateurs de ses
contraintes. Sauvegardés dans un même fichier, un modèle et son maillage
restent liés à la lecture : le modèle relu est posé sur le maillage relu. Le
motif des matrices n'est pas écrit ; il se recalcule au premier assemblage.

## Côté Rust

`Model` est l'agrégat de `Handle<SubModel>`. Il garde aussi le motif de chaque
genre de matrice, dans `matrix_patterns`, un tableau de
`OnceLock<Arc<AssemblyPattern>>` à une case par `MatrixKind`. Son `check_push`
refuse un sous-modèle d'un autre `Coords`, et son `post_push` vide les motifs.
`ops::matrix::assemble_kind` demande ses blocs à chaque sous-modèle, puis
prend le motif du modèle, construit au premier appel par
`ops::scatter::build_pattern`, et y répartit les valeurs en parallèle.

`SubModel` est une `enum` à une variante par physique, chacune enveloppant la
struct de `src/models/` qui porte ses données. L'`enum` sert à ranger et à
sauvegarder : `bincode` écrit le rang de la variante, donc une nouvelle
variante se place à la fin. Le code commun (l'agrégat, l'assemblage,
l'affichage) passe par `as_kind()`, qui rend la struct de la variante en
`&dyn SubModelKind`.

`SubModelKind` (`src/models/mod.rs`) est ce que toute physique déclare : ses
variables, ses natures (`physics()`, un `&'static [Physics]`), son nom
(`label()`), ses blocs pour chaque genre de matrice, ses termes de forces
internes et externes, son affichage. Trois traits la complètent selon la
physique, rendus par `as_domain`, `as_constraint` et `as_behavior` (`None` par
défaut) : `Domain` pour le matériau et les matrices élémentaires, `Constraint`
pour les nœuds-multiplicateurs et les relations, `Behavior` pour la loi de
comportement. Le chapitre [Ajouter une physique](ajouter-une-physique.md)
détaille ces traits. Un bloc de matrice monté à la main reçoit ses natures par
`SubMatrix::set_physics`, que Python n'expose pas.

```rust,ignore
{{#include ../../tests/doc_conventions.rs:modele}}
```
