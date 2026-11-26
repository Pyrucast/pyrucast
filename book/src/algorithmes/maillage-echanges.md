# Lire et échanger des maillages

## Lecture d'un maillage gmsh : `read_gmsh`

{{#pyalgo mesh.read_gmsh mesh.read_gmsh_str}}

`read_gmsh` importe un maillage produit par **gmsh** (fichier `.msh`,
versions **MSH 2.2** et **MSH 4.1**, en **ASCII comme en binaire** —
l'endianness est lue dans le fichier). L'appelant fournit la `Coords` dans
laquelle lire (il garde ainsi la main sur les nœuds) ; le résultat est un
`dict` Python qui associe à **chaque groupe physique** son `Mesh` :

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-read_gmsh}}
```

### Groupes, types et `Coords` partagée

- **Un `Mesh` par groupe physique**, et **un sous-maillage par type
  d'élément** à l'intérieur de chaque groupe. Les éléments sans groupe
  physique sont rangés sous la clé `"<ungrouped>"`.
- **Tous les `Mesh` partagent la `Coords` fournie** : un nœud à la frontière
  de deux groupes (p. ex. un nœud du bord `"bottom"` qui appartient aussi à
  la surface `"plate"`) est **le même nœud** des deux côtés — pas un
  doublon. Comme c'est *votre* `Coords`, vous gardez le handle pour poser
  des conditions aux limites sur une région nommée lue dans le fichier :

  ```python
  coords = pyrucast.Coords(dim=2)
  regions = pyrucast.mesh.read_gmsh(coords, "piece.msh")
  plate = regions["plate"]
  bottom = regions["bottom"]  # même Coords que plate

  fes = pyrucast.FiniteElementSpace(plate)
  # ... assemblage sur 'plate', blocage des nœuds de 'bottom', etc.
  ```

  La `Coords` peut déjà contenir de la géométrie : l'import s'y **ajoute**.
  Si vous égarez le handle côté Python, `mesh.coords()` le récupère depuis
  n'importe quel `Mesh` du `dict`.

### Types d'éléments reconnus

Les codes gmsh sont traduits vers les types pyrucast ; l'ordre local des
nœuds coïncide déjà avec le repère de référence, la connectivité est donc
copiée telle quelle.

| Code gmsh | Type pyrucast |
|---|---|
| `1`  | `SEG2` |
| `2`  | `TRI3` |
| `3`  | `QUA4` |
| `4`  | `TET4` |
| `5`  | `HEX8` |
| `6`  | `PENTA6` |
| `7`  | `PYRA5` |
| `15` | `POI1` |
| `8`  | `SEG3` |
| `9`  | `TRI6` |
| `16` | `QUA8` |
| `10` | `QUA9` |
| `11` | `TET10` |
| `17` | `HEX20` |
| `18` | `PENTA15` |
| `12` | `HEX27` |

Pour les types quadratiques volumiques (`TET10`, `HEX20`, `PENTA15`, `HEX27`),
gmsh numérote les nœuds de milieu d'arête (et de face pour `HEX27`) dans un
ordre différent de la convention pyrucast (VTK) : la connectivité est
**réalignée** à la lecture (même permutation que meshio). Tout autre type gmsh
(ordre 3 et plus…) lève une erreur explicite.

### Dimension

gmsh stocke toujours trois coordonnées par nœud ; c'est la **dimension de la
`Coords` fournie** qui décide combien sont conservées. On lit donc dans une
`Coords(dim=2)` pour aplatir un maillage planaire sur `xy`, ou dans une
`Coords(dim=3)` pour garder le relief.

> Seuls les nœuds **référencés** par un élément sont matérialisés dans la
> `Coords` ; les nœuds isolés listés mais utilisés par aucun élément sont
> ignorés.

`read_gmsh_str(coords, text)` fait la même chose à partir du **texte** du
fichier déjà chargé en mémoire (utile pour les tests ou un `.msh` reçu sur
le réseau). Côté Rust, `ops::mesh::read_gmsh(coords, path)` et
`ops::mesh::read_gmsh_str(coords, text)` renvoient un
`Vec<(String, Mesh)>` ordonné.

## Déléguer le maillage à gmsh : `from_gmsh`

{{#pyalgo mesh.from_gmsh}}

Les mailleurs de pyrucast couvrent le cas courant ; les géométries difficiles,
elles, se **délèguent**. On définit la géométrie et on maille dans gmsh — c'est
son métier —, puis on appelle `from_gmsh` pour récupérer le résultat, **sans
passer par un fichier** :

```python
{{#include ../../../tests/python/test_doc_gmsh.py:mesh-from_gmsh}}
```

pyrucast **lit** gmsh, il ne le pilote pas : aucune fonction pyrucast ne crée de
géométrie ni ne lance de maillage. `from_gmsh` rend un couple
`(maillages, champs)`. Les maillages forment le même `dict[str, Mesh]` que
`read_gmsh`, avec les mêmes règles — un `Mesh` par groupe physique, une zone par
type d'élément, une seule `Coords` partagée dont la dimension décide combien des
trois coordonnées de gmsh sont gardées, et `"<ungrouped>"` pour le reste.

Les **surfaces et les points nommés** dans gmsh (`addPhysicalGroup(…, name=…)`)
deviennent donc les clés du dictionnaire, et c'est sur elles qu'on pose ensuite
les conditions aux limites. gmsh maille ses entités ponctuelles : un point nommé
arrive comme un `Mesh` POI1.

### Les vues gmsh deviennent des champs

Les **vues** du modèle (post-traitement gmsh, `gmsh.view.addModelData`) sont
lues aussi, sauf si l'on passe `views=False` :

| vue gmsh | champ pyrucast |
|---|---|
| `NodeData` | `NodeField` |
| `ElementData` | `ElementField` à **un** point par maille, avec une zone sur chaque groupe dont il définit toutes les mailles |
| plusieurs pas de temps | `Evolution` sur les temps de la vue |

Une vue scalaire garde son nom comme nom de composante ; une vue à `n`
composantes les nomme `nom_0 … nom_{n-1}`. Les autres sortes de vues
(`ElementNodeData`, vues « liste ») sont ignorées avec un avertissement.

### Ce qui est copié, et ce qui ne l'est pas

La chaîne a trois maillons, dont deux sont gratuits :

| maillon | copie ? | pourquoi |
|---|---|---|
| gmsh → numpy | non | les tableaux que rend l'API gmsh sont des **vues** sur la mémoire de gmsh, libérées au ramasse-miettes du tableau |
| numpy → pyrucast | non | lecture par le **protocole tampon** — d'où le plancher Python 3.11 du projet, `PyObject_GetBuffer` n'étant entré dans l'API limitée qu'à cette version. Les étiquettes `uint64` de gmsh comme `int64` de medcoupling sont empruntées telles quelles |
| construction du maillage | **oui, une passe** | une `Coords` et un sous-maillage possèdent leurs tableaux : les coordonnées sont écrites directement dans la `Coords`, chaque connectivité est allouée une fois à sa taille et réécrite **sur place** dans l'ordre pyrucast |

Autrement dit : une vue jusqu'à la frontière Rust, puis **une seule passe**.
Rien n'est jamais matérialisé élément par élément côté Python. Une fois l'appel
rendu, pyrucast possède ses données — `gmsh.finalize()` ne les emporte pas.

> Un objet sans tampon à prêter — une `list`, ce que gmsh rend quand numpy n'est
> pas installé — est lu par la conversion ordinaire. Le résultat est le même,
> au prix d'une copie de plus.

Un tableau contigu se lit sans copie quelle que soit sa forme, à plat, dans
l'ordre de sa mémoire : une table `(nœuds, 3)` donne le même import que son
`ravel()`, une connectivité `(mailles, nœuds par maille)` aussi.

### Restreindre l'import

`dim` limite l'import à une dimension, `tag` à une seule entité de cette
dimension (`from_gmsh(coords, dim=2, tag=1)`). La table des nœuds est lue en
entier quoi qu'il arrive — une maille de surface s'appuie sur des nœuds classés
sur ses courbes de bord — et seuls les nœuds **référencés** sont matérialisés.

> Une nuance à connaître pour le contrôle croisé : `gmsh.write()` n'écrit par
> défaut que les éléments portant un groupe physique, alors que `from_gmsh` voit
> **tout** le modèle. Le `.msh` relu peut donc contenir moins que l'import
> direct — la différence tient dans `"<ungrouped>"`.

## Lire un fichier MED : `from_medcoupling`

{{#pyalgo mesh.from_medcoupling}}

Le format **MED** est celui de Salome et de code_aster. pyrucast le lit et
l'écrit au travers de **medcoupling**, la bibliothèque MED de Salome
(`pip install medcoupling`). Elle n'est **pas** une dépendance de pyrucast :
`import pyrucast` ne la charge pas, seul l'appel de `from_medcoupling` (ou de
`to_medcoupling`) l'importe.

```python
{{#include ../../../tests/python/test_doc_medcoupling.py:mesh-from_medcoupling}}
```

La source est un chemin vers un `.med`, un `medcoupling.MEDFileData` ou un
`medcoupling.MEDFileUMesh` (maillage seul) ; `mesh_name` choisit un maillage du
fichier (par défaut le premier). Le résultat est un couple
`(maillages, champs)` :

- **un `Mesh` par groupe MED** — groupes de mailles de tous les niveaux, et
  groupes de nœuds sous forme de `Mesh` POI1 —, tous sur la `Coords` fournie,
  les mailles sans groupe sous `"<ungrouped>"`. Les **familles** MED, qui
  codent l'appartenance aux groupes, sont traduites une fois par famille et
  par type, jamais maille par maille ;
- **les champs**, sous leur nom MED :

| champ MED | champ pyrucast |
|---|---|
| `ON_NODES` | `NodeField` sur les nœuds définis (profil compris) |
| `ON_CELLS` | `ElementField` à **un** point par maille |
| `ON_GAUSS_PT` | `ElementField` sur les points de Gauss de pyrucast — si la règle MED est la même, sinon une erreur explicite |
| plusieurs pas de temps | `Evolution` sur les temps du champ |

Un champ aux mailles reçoit une zone sur chaque sous-maillage de groupe dont il
définit **toutes** les mailles. Les autres discrétisations (`ON_GAUSS_NE`…)
sont ignorées avec un avertissement.

La **numérotation MED** des nœuds dans une maille diffère de celle de pyrucast
(qui est celle de VTK) pour les volumes : MED parcourt la première face dans
l'autre sens. La permutation est une propriété de l'élément
(`ElementKind::med_permutation`) et s'applique en Rust, pendant la passe de
construction. Elle est vérifiée contre medcoupling lui-même : pour chacun des
quinze types, la maille de référence écrite en MED a son volume **positif** et
ses nœuds milieux au milieu des arêtes que medcoupling en déduit.

> medcoupling n'est publié que pour **Linux x86_64 et Windows** (CPython 3.9 à
> 3.13) : pas de macOS ni d'ARM. Sa licence est la LGPL, ce qui convient à une
> dépendance facultative importée à l'exécution.

## L'opérateur en dessous : `from_arrays`

{{#pyalgo mesh.from_arrays}}

`from_gmsh` et `from_medcoupling` sont les seules fonctions de `pyrucast.mesh`
écrites en Python : elles ont besoin d'un interpréteur portant le module
`gmsh` ou `medcoupling`, ce que Rust ne peut pas avoir. Elles ne font
qu'aller chercher des tableaux et les passer à `from_arrays`, l'**import
générique** que partagent tous les formats — utilisable directement quand on
tient déjà les tableaux :

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-from_arrays}}
```

Le format est commun à tous les échanges :

- **nœuds** — une étiquette entière par nœud (`uint64` ou `int64`) et 1 à 3
  coordonnées par nœud ;
- **blocs** — `(type, connectivité, étiquettes de mailles, groupes)` : un bloc
  réunit un type d'élément et **une combinaison de groupes**, soit
  exactement une famille MED ou une entité gmsh. Les étiquettes de mailles ne
  servent qu'aux champs aux mailles (une séquence vide sinon, ou un triplet
  sans elles) ;
- **champs** — `node_fields=[(composantes, étiquettes de nœuds, valeurs)]`,
  `cell_fields=[(composantes, étiquettes de mailles, valeurs, disposition)]`,
  la disposition valant `"cell"` ou une liste de règles de Gauss
  `(type, nœuds de référence, points, poids)` ;
- **ordre** — `order="pyrucast"`, `"gmsh"` ou `"med"` : la numérotation des
  nœuds dans une maille, réalignée en Rust.

`from_arrays` rend `(maillages, champs aux nœuds, champs aux éléments)`. Le
fichier `.msh` passe par le même moteur : le regroupement, le partage des
nœuds et les permutations ne sont écrits qu'une fois, et un test compare maille
pour maille ce que le fichier et la mémoire rendent de la même géométrie.

Côté Rust, `ops::mesh::from_arrays(coords, node_tags, node_coords, blocks,
node_values, cell_values, order)` prend des `CellBlock`, `NodeValues` et
`CellValues` qui ne font qu'**emprunter** les tableaux, et rend un `Imported`.
Pour un maillage de 884 000 hexaèdres, la construction prend 79 ms contre
498 ms pour l'ancien import gmsh (`cargo bench --bench mesh -- from_arrays`) :
plus de table de hachage par nœud, plus d'allocation ni de nom de groupe par
maille.

### Le chemin inverse : `to_arrays`

{{#pyalgo export.to_arrays}}

`pyrucast.export.to_arrays` rend la même forme, à partir de maillages et de
champs pyrucast — c'est la sortie que partagent l'export VTK, `to_gmsh` et
`to_medcoupling` :

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:export-to_arrays}}
```

- les nœuds sont numérotés dans l'ordre de première apparition, à partir de
  `first_tag` (1 par défaut, 0 pour medcoupling) ;
- une maille présente dans plusieurs groupes — ce que produit tout import gmsh
  ou MED — n'est écrite **qu'une fois**, dans un bloc qui porte tous ses
  groupes ;
- un champ aux nœuds donne une ligne par nœud, **0** là où il ne définit rien ;
  un champ aux éléments donne la **moyenne** de ses points de Gauss par maille,
  ou avec `gauss=True` ses valeurs brutes et la règle de chaque type ;
- chaque tableau est un `pyrucast.Array`, en lecture seule, que
  `numpy.asarray` ou `memoryview` lisent **sans copie** — sans que numpy soit
  une dépendance de pyrucast.

Deux traductions servent aux adaptateurs et restent utilisables seules : le
type pyrucast d'un code gmsh, et la règle de Gauss de pyrucast exprimée dans
l'élément de référence d'un autre format (et l'appariement inverse, qui refuse
une règle différente) :

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:gauss_codes}}
```
