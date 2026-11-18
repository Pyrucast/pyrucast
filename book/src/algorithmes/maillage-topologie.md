# Bords, orientation et sélections

## Bord d'une surface : `border`

{{#pyalgo mesh.border}}

`border(mesh, angle_deg=None)` est l'**inverse** de `triangulate_surface` : il
prend un maillage de surface (cellules `TRI3` / `QUA4`) et renvoie son **bord**
sous forme de boucles `SEG2` fermées. Une surface quadratique (`TRI6`,
`QUA8`, `QUA9`) donne des boucles `SEG3`, qui reprennent le nœud milieu de
chaque arête de bord ; une surface qui mêle zones linéaires et quadratiques
est refusée.

Une arête de cellule utilisée par **exactement une** cellule est une arête de
bord ; les arêtes intérieures sont partagées par deux cellules (orientations
opposées) et s'annulent. Les arêtes de bord de **tous** les sous-maillages de
surface sont regroupées — la sortie `QUA4` + `TRI3` de `triangulate_surface` donne donc un
bord commun unique — puis chaînées en boucles fermées.

Le résultat est un `Mesh` avec **un sous-maillage SEG2 par boucle** : une seule
boucle pour un domaine simplement connexe, plusieurs quand le domaine a des
trous ou des morceaux disjoints. Chaque boucle garde
l'orientation CCW du bord (boucle extérieure CCW, trous CW) : le résultat peut
donc **réalimenter directement** `triangulate_surface`. Les nœuds d'origine
sont réutilisés (et re-référencés).

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-border}}
```

### Découpe par angle (`angle_deg`)

Avec un `angle_deg`, chaque boucle est en plus **découpée en arêtes ouvertes**
à ses **coins** — le pendant 1D du découpage en faces planes de
[`skin`](#peau-dun-volume--skin). Un nœud est un coin quand le bord y **tourne**
de plus de `angle_deg` degrés (l'angle entre les directions des arêtes entrante
et sortante). Chaque arête — une suite maximale de segments quasi alignés entre
deux coins — devient son propre sous-maillage `SEG2` (un côté droit d'un carré,
même subdivisé, reste **une** arête). Une boucle sans aucun coin (bord courbé
dont tous les virages restent sous le seuil) est conservée comme une boucle
fermée. `angle_deg=None` (défaut) garde chaque bord en une boucle fermée.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:border_angle}}
```

Les sous-maillages POI1 (un point n'a pas d'arête) sont ignorés. La fonction
lève une erreur si le maillage n'a aucune cellule de surface, s'il porte des
cellules autres que POI1/TRI3/QUA4/TRI6/QUA8/QUA9 (les bords 1D et 3D ne sont pas gérés ici —
voir [`skin`](#peau-dun-volume--skin) pour le bord d'un volume), ou si le bord
n'est pas un ensemble propre de boucles fermées (arête ouverte ou non-manifold).

Côté Rust, `ops::mesh::border(&mesh, angle_deg)`.

## Peau d'un volume : `skin`

{{#pyalgo mesh.skin}}

`skin(mesh, angle_deg=None)` est le pendant **3D** de `border` : il prend un
maillage volumique et renvoie sa **peau** — la surface extérieure — découpée en
**faces planes**, un sous-maillage par face.

Il accepte **tous** les types volumiques : `TET4`, `PYRA5`, `PENTA6`, `HEX8` et
leurs homologues quadratiques `TET10`, `PENTA15`, `HEX20`, `HEX27`. Chaque
élément déclare lui-même ses facettes, si bien qu'aucun type n'est laissé de
côté.

Une facette d'élément volumique (une face de `TET4`, de `HEX8`, …) utilisée par
**exactement une** cellule est une facette de bord ; les facettes intérieures
sont partagées par deux cellules et s'annulent. Les facettes de bord de **tous**
les sous-maillages volumiques sont regroupées, puis réparties en **faces
planes** : deux facettes adjacentes (partageant une arête) appartiennent à la
même face tant qu'elles restent **quasi coplanaires** — l'angle entre leurs
normales sortantes est inférieur ou égal à `angle_deg` degrés (défaut **1°**).

**Une facette est émise dans son propre type** : un `HEX8` donne des `QUA4`,
un `TET10` donne des `TRI6`, un `HEX27` des `QUA9`. La peau d'un maillage
quadratique est donc elle-même quadratique et **conserve ses nœuds milieux** —
on peut la remailler ou poser un chargement dessus sans perdre le degré.
L'appariement des facettes se décide sur leurs **coins**, ce sur quoi deux
mailles voisines s'accordent quel que soit leur degré.

Chaque groupe devient un sous-maillage par type de facette (une face mêlant
triangles et quadrangles, p. ex. à une interface `PYRA5`/`HEX8`, en produit un
de chaque). Un cube donne ainsi six sous-maillages, un prisme cinq (deux
chapeaux triangulaires, trois flancs quadrangulaires), une pyramide cinq (la
base carrée et quatre triangles). Les facettes conservent leur orientation
sortante ; les nœuds d'origine sont réutilisés (et re-référencés).

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-skin}}
```

Un `angle_deg` plus grand regroupe des faces plus courbées (un cylindre facetté
devient une seule paroi) ; un `angle_deg` proche de 0 isole chaque facette. Les
sous-maillages POI1 sont ignorés. La fonction lève une erreur si le maillage n'a
aucune cellule volumique, s'il porte des cellules de dimension topologique
inférieure (surface, ligne), ou si l'espace n'est pas 3D.

Côté Rust, `ops::mesh::skin(&mesh, angle_deg)`.

## Orientation des cellules : `orient` et `invert`

{{#pyalgo mesh.orient mesh.invert}}

`orient(mesh)` (Cast3M `ORIE`) **harmonise** l'orientation des cellules d'un
maillage, et `invert(mesh)` (Cast3M `INVE`) l'**inverse**. Les deux travaillent
en **toute dimension** — segments `SEG*` (1D), faces `TRI*`/`QUA*` (2D),
volumes `TET*`/`PENTA*`/`HEX*` (3D), variantes linéaires **et quadratiques** — et
renvoient un **maillage neuf** qui reflète l'entrée sous-maillage par
sous-maillage (mêmes types, mêmes couleurs, mêmes nœuds partagés) ; l'entrée est
laissée intacte.

### Cadre unifié : facettes orientées

Le bord orienté d'une cellule est une somme signée de ses **facettes de
codimension 1** :

- `SEG*` (d = 1) : les deux nœuds extrémité — queue (`−1`) et tête (`+1`) ;
- `TRI*` / `QUA*` (d = 2) : les arêtes orientées ;
- `TET*` / `PENTA*` / `HEX*` (d = 3) : les faces orientées sortantes.

Chaque occurrence se réduit à un couple `(clé, signe)` où `clé` est la liste
**triée** des nœuds (coins) de la facette et `signe ∈ {−1, +1}` encode son
orientation par rapport à une orientation canonique de la clé. **Deux cellules
partageant une facette sont cohérentes ssi elles lui donnent des signes
opposés.** Les clés de dimensions différentes (1 / 2 / ≥ 3 nœuds) ne se
confondent jamais : un maillage mixte se sépare en composantes par dimension.

### `orient` — cohérence, pas de sens absolu

`orient` propage une orientation cohérente à travers les facettes partagées par
un parcours en largeur du graphe dual. Chaque **composante connexe** est amorcée
par sa cellule d'indice le plus bas, qui **garde** son orientation (choix
déterministe, reproductible bit à bit) ; les autres sont retournées au besoin.

`orient` **ne choisit pas** de sens absolu « sortant » : pour une surface fermée
il laisse le tout entièrement sortant *ou* entièrement rentrant selon la graine.
Pour choisir le sens (p. ex. définir l'intérieur d'un trou), composer avec
`invert`. Les facettes **non-manifold** (partagées par plus de deux cellules)
n'imposent aucune contrainte et sont ignorées.

### `invert` — retournement inconditionnel

`invert` applique à **chaque** cellule sa permutation d'inversion
(`ElementType::reversal_permutation`, la réflexion échangeant les deux premiers
axes de référence — pour `SEG*` la négation de l'axe unique). Les `POI1` (sans
orientation) sont inchangés. Appliqué deux fois, `invert` redonne le maillage de
départ.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-orient}}
```

Côté Rust, `ops::mesh::orient(&mesh)` et `ops::mesh::invert(&mesh)`.

## Mise en chaîne d'une ligne : `chain`

{{#pyalgo mesh.chain}}

`orient` fait *pointer* les segments d'une courbe dans le même sens, mais les
laisse là où ils sont dans la connectivité. `chain(mesh)` est le complément :
il les met dans l'**ordre du parcours**, de sorte que lire les mailles l'une
après l'autre revient à marcher le long de la courbe d'un bout à l'autre.

```text
  5 6                    1 2
  1 2      chain  →      2 3
  3 4                    3 4
  2 3                    4 5
  4 5                    5 6
```

Chaque sous-maillage est chaîné **indépendamment** (il garde ses mailles, son
type et sa couleur de face) et doit former **une seule chaîne continue** :
chaque nœud porte un ou deux segments, et les mailles constituent **un seul**
morceau connexe, ouvert (deux extrémités libres) ou refermé en boucle. Tout le
reste est une **erreur** — un nœud à trois segments (embranchement), ou
plusieurs morceaux disjoints. Il faut alors découper l'entrée en un
sous-maillage par branche.

Les mailles sont **retournées** au besoin en chemin (le nœud milieu d'un `SEG3`
reste au milieu) : `chain` oriente donc aussi, inutile de passer par `orient`
d'abord.

### Par où la chaîne commence

Une chaîne ouverte se lit depuis une extrémité libre. Si **exactement une** des
deux est déjà la **queue** de son segment, c'est elle qui amorce le parcours :
une courbe déjà orientée de façon cohérente **garde son sens**, et `chain` se
réduit à une permutation des mailles. Sinon (les deux extrémités sont des
queues, ou aucune) c'est celle de plus petit numéro de nœud, ce qui rend le
résultat déterministe. Une boucle fermée n'a pas d'extrémité libre : elle part
de son plus petit numéro de nœud, dans le sens du segment qui l'a déjà pour
queue.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-chain}}
```

Côté Rust, `ops::mesh::chain(&mesh)` — ou la méthode `mesh.chain()`.

## Éléments s'appuyant sur des nœuds : `elements_on`

{{#pyalgo mesh.elements_on}}

`elements_on(mesh, points, strict=True)` renvoie le **sous-ensemble** des
éléments de `mesh` qui **s'appuient** sur les nœuds de `points` — l'opérateur
historique `ELEM … APPUYE`. Seul l'**ensemble des nœuds référencés** par
`points` compte (typiquement un maillage de points POI1) ; ni le type ni la
connectivité de `points` n'importent.

Le critère dépend de `strict` :

- `strict=True` — on garde une cellule lorsque **tous** ses nœuds sont dans
  l'ensemble (`APPUYE STRICTEMENT`) ;
- `strict=False` — on garde une cellule dès qu'**au moins un** de ses nœuds y
  est (`APPUYE`).

Le résultat **épouse la structure** de `mesh` sous-maillage par sous-maillage
(même ordre, mêmes types d'éléments, mêmes couleurs) : chaque sous-maillage de
sortie porte les cellules retenues du sous-maillage d'entrée correspondant,
**éventuellement vide**. Les zones restent séparées (jamais de fusion). Au
besoin, `mesh.consolidate(mesh)` élimine ou fond ensuite les zones vides ou
redondantes. Les cellules retenues réutilisent les nœuds d'origine (refcount
incrémenté) ; `mesh` est laissé intact.

Les deux maillages doivent vivre sur la **même `Coords`** (un identifiant de
nœud n'a de sens qu'au sein d'une `Coords`), sinon une erreur est levée. Un
`points` vide ne retient rien.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-elements_on}}
```

Côté Rust, `ops::mesh::elements_on(&mesh, &points, strict)`.

## Intersection et différence : `intersection`, `difference`

{{#pyalgo mesh.intersection mesh.difference}}

`intersection(a, b)` garde les mailles de `a` que `b` contient aussi, et
`difference(a, b)` celles que `b` ne contient pas. Une maille est la même quand
elle a le même type et les mêmes nœuds, **dans le même ordre**. Un triangle
numéroté à partir d'un autre sommet, ou retourné, est donc une autre maille :
ses points de Gauss et sa normale ne sont pas les mêmes.

Le résultat épouse la structure de `a`, zone par zone (même ordre, mêmes types
d'éléments, mêmes couleurs). Chaque zone garde, dans l'ordre de `a`, celles de
ses mailles qu'une zone de `b` du même type contient (ou ne contient pas), et
reste à sa place même vide. Les deux opérations se partagent ainsi chaque zone
de `a`.

Une zone gardée en entier revient **telle quelle** : c'est le même objet, pas
une copie. C'est le cas quand `b` contient cette zone elle-même ou chacune de
ses mailles et, pour une différence, quand `b` n'en contient aucune. Un champ
posé sur cette zone se combine alors avec un champ posé sur le résultat.

Deux maillages sur deux `Coords` différents n'ont aucune maille commune : un
numéro de nœud n'a de sens qu'au sein d'un même `Coords`.

Les opérateurs historiques `INTE` et `DIFF` reconnaissent une maille à ses
nœuds pris dans n'importe quel ordre, et `DIFF` rend la différence symétrique :
les mailles que l'un des deux maillages a et que l'autre n'a pas. Elle s'écrit
ici `difference(a, b) | difference(b, a)`.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-intersection}}
```

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-difference}}
```

Côté Rust, `ops::mesh::intersection(&a, &b)` et `ops::mesh::difference(&a, &b)`,
ou les méthodes `a.intersection(&b)` et `a.difference(&b)`.

## Sélection de nœuds par région géométrique : la famille `points_*`

{{#pyalgo mesh.points_in_sphere mesh.points_on_sphere mesh.points_on_plane mesh.points_below_plane mesh.points_on_line mesh.points_in_cylinder mesh.points_on_cylinder mesh.points_in_cone mesh.points_on_cone mesh.points_in_torus mesh.points_on_torus}}

Pour poser une condition aux limites il faut d'abord **désigner des nœuds**.
La famille `points_*` répond à cette question par une **région géométrique** —
l'équivalent de l'opérateur historique `POIN … PLAN / DROIT / CYLI / SPHE` :

```text
points_<in|on|below>_<forme>(mesh, …géométrie…, tol=None) -> Mesh POI1
```

Toutes ces fonctions renvoient un **maillage POI1 calqué sur l'entrée** : un
sous-maillage par sous-maillage de `mesh`, dans le même ordre, **éventuellement
vide**. La sélection conserve donc le zonage de sa source — on sait de *quelle*
zone vient chaque nœud, et on peut travailler zone par zone en indexant le
résultat. `mesh.consolidate(sel)` retombe sur une zone POI1 unique quand ce découpage ne
sert pas.

Les nœuds sont **dédoublonnés dans l'ordre de première apparition** dans la
connectivité, exactement comme [`to_poi1`](maillage.md) — une sélection totale
reproduit `to_poi1` nœud pour nœud.

### `in` et `on`

Deux familles, deux lectures :

- `points_in_*` — **dans** la région fermée, élargie de `tol` ;
- `points_on_*` — à moins de `tol` de la **surface** de la région, des deux
  côtés.

Deux formes sont **fermées par des faces planes**, et la distinction compte :
`points_on_cylinder` et `points_on_cone` ne retiennent que la **surface
latérale**, pas les disques d'extrémité — ceux-ci sont plats, c'est
`points_on_plane` qui les coupe. Le tore, lui, est une surface fermée : la
question ne se pose pas.

Le plan n'a pas de « dedans » : il a deux côtés, d'où `points_below_plane`, le
demi-espace **opposé** à la normale (plan compris). Il n'existe pas de
`points_above_plane` : retourner la normale donne l'autre moitié.

Enfin, la droite est **infinie** là où le cylindre est **borné** : pour une
sélection le long d'un axe mais limitée au segment, c'est `points_in_cylinder`
avec un petit rayon.

### La tolérance `tol`

`tol` est la **précision géométrique** du test, mesurée comme une **distance à
la surface** de la région. `tol=None` demande la valeur par défaut :
`1e-6 ×` la diagonale de la boîte englobante du maillage. C'est ce qui rend les
opérateurs **sans échelle** — le même appel marche sur une équerre en
millimètres et sur un barrage en kilomètres — et ce qui rend `points_on_plane`
utilisable sur des nœuds sortis d'un mailleur plutôt que d'une arithmétique
exacte.

Pour le cône, la distance est prise **perpendiculairement** à la surface
inclinée, et non radialement : la bande reste large de `tol` quelle que soit la
pente.

### Le cas « un seul nœud »

La requête du **nœud le plus proche** d'un point ne peut en renvoyer qu'un ;
elle n'est donc pas dans cette famille et ne renvoie pas de POI1, mais un
`Node` : c'est la méthode `mesh.nearest_node([x, y])`, des deux côtés — voir
[Opérateurs géométriques](geometrie.md).

### Repère de travail

Les nœuds sont testés dans les coordonnées où ils sont **stockés**. En
[axisymétrie](../coords.md), c'est le demi-plan méridien `(r, z)` et non le
solide de révolution : une « sphère » y est un cercle du méridien. Le tore,
qui a besoin d'un axe hors du plan pour être un tore, est **3D seulement**.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:selections}}
```

En 3D, les formes de révolution sélectionnent alésages, arbres et gorges :

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:selections_courbes}}
```

Côté Rust, `ops::mesh::points_on_plane(&mesh, &origin, &normal, tol)` et
consorts, avec `tol: Option<f64>`.

## Soudure des nœuds proches : `merge_nodes`

{{#pyalgo mesh.merge_nodes}}

`merge_nodes(mesh, tol)` **soude** entre eux les nœuds distants de moins de
`tol` (distance euclidienne). C'est l'opération de « recollage » classique :
quand deux morceaux maillés séparément se rejoignent le long d'une interface,
leurs nœuds y sont colocalisés mais **distincts** ; `merge_nodes` les fond en
un seul, rendant le maillage topologiquement connexe.

Chaque cluster de nœuds proches est représenté par **un seul** nœud — celui de
plus petit identifiant —, et ce représentant **garde ses propres coordonnées**
(aucune moyenne : on ne déplace jamais la géométrie en douce). La connectivité
de chaque sous-maillage est réécrite pour pointer vers les représentants ; la
structure de sous-maillages (types, ordre, couleurs) est préservée.

Un cluster est une **composante connexe** de la relation « distants de moins de
`tol` » : la soudure se propage de proche en proche. Sur une chaîne a—b—c où a
et b se touchent, b et c aussi, mais a et c non, les trois n'en font qu'un.
C'est aussi pourquoi une tolérance de l'ordre de la taille de maille effondre
toute une zone au lieu d'une seule interface : `tol` se choisit petit devant
l'arête la plus courte, et la ligne de bilan est là pour le vérifier.

**Coût.** La recherche des voisins passe par une grille uniforme dont la maille
ne descend jamais sous `tol` : chaque nœud ne visite que les cases que sa boule
de rayon `tol` touche vraiment — une seule, dans le cas courant — et cette
phase est parallélisée. Compter une poignée de secondes pour dix millions de
mailles, la mémoire de travail restant de l'ordre de quelques octets par nœud.

Une cellule qui **s'effondre** — c'est-à-dire qui référence deux fois le même
représentant après soudure (un `SEG2` dont les deux bouts fusionnent, un `TRI3`
à deux coins confondus, …) — est **abandonnée** : elle est dégénérée. Les
cellules `POI1` (un seul nœud) ne s'effondrent jamais et sont toujours
conservées ; dédupliquer des points colocalisés reste le rôle de
`mesh.consolidate`, pas celui-ci.

`tol` doit être ≥ 0 ; `tol = 0` ne soude que les nœuds **exactement**
colocalisés. Seuls les nœuds **référencés** par le maillage sont concernés. Le
maillage d'entrée est laissé intact ; les nœuds soudés disparaissent de la
connectivité du résultat et deviennent récupérables par le GC de la `Coords`
une fois plus rien ne les référence.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-merge_nodes}}
```

**Bilan à l'écran.** Chaque appel imprime une ligne sur la sortie standard —
nœuds soudés, mailles supprimées, tolérance employée :

```text
merge_nodes: 12 node(s) welded, 3 cell(s) dropped, tol = 0.000001
merge_nodes (in place): 12 node(s) welded, cells untouched, tol = 0.000001
```

C'est une étape qu'on veut voir passer dans un journal de construction :
`tol` est un pari sur la géométrie, et cette ligne est ce qui dit s'il était
bon. En place, la ligne ne parle pas de mailles supprimées : il ne peut pas y
en avoir, l'opérateur refuse plutôt (voir plus bas).

> `merge_nodes` opère **au sein d'une même `Coords`** (l'invariant du `Mesh`
> impose déjà une `Coords` commune à tous les sous-maillages). Deux pièces
> maillées dans des `Coords` séparées ne se soudent donc pas : il faut d'abord
> les amener dans la même `Coords`.

### Souder sur place : `in_place=True`

Par défaut `merge_nodes` **copie** : il rend un maillage neuf et laisse ses
entrées intactes. Les maillages d'origine, eux, gardent donc leurs nœuds
dupliqués — ce qui oblige à ne plus manipuler qu'un troisième maillage, et à
recâbler tout ce qui pointait vers les deux premiers.

`merge_nodes(mesh, tol, in_place=True)` réécrit à la place la connectivité des
sous-maillages **existants** — effet de bord assumé et voulu — et renvoie le
maillage lui-même. Comme l'union `mesh_a | mesh_b` **partage** les
sous-maillages (elle ne les copie pas), souder l'union soude du même coup
`mesh_a` et `mesh_b` :

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:merge_in_place}}
```

Ce que la mutation ne touche pas : **la structure du maillage**. Mêmes
sous-maillages, mêmes types, même nombre de cellules dans le même ordre — seul
*quel nœud* une cellule référence change. C'est ce qui rend l'effet de bord
tenable : tout indice déjà détenu sur ces sous-maillages (numéros de cellules,
et donc les champs par élément qui s'appuient dessus) reste valide. Les
coordonnées des nœuds ne bougent pas non plus.

D'où deux refus, vérifiés **sur tout le maillage avant la moindre écriture**
(un appel rejeté ne modifie donc rien) :

- une cellule qui **s'effondrerait** est une **erreur**, là où la variante
  copiante l'abandonne : l'abandonner changerait le nombre de mailles, c'est-à-
  dire précisément l'invariant sur lequel repose l'appel sur place. Baissez
  `tol`, ou passez par la variante copiante ;
- un sous-maillage **scellé** est une erreur : un espace d'éléments finis, une
  matrice — ou un champ, si c'est une zone POI1 qui lui sert de support — l'a
  capturé et lit sa numérotation de nœuds. Soudez avant de les construire (ou
  repartez d'un `duplicate()`).

Les caches dérivés de la connectivité (index des nœuds, compagnon POI1 de
`to_poi1`) sont invalidés par la réécriture, et les refcounts suivent : chaque
emplacement réécrit incrémente son nouveau nœud et décrémente l'ancien.

Le retour est **exactement le maillage passé** — les mêmes sous-maillages,
dont l'intérieur a changé —, pas une copie : en Python, `out is mesh`. On peut
donc l'ignorer, ou chaîner dessus, au choix.

Côté Rust, c'est le **même** opérateur avec le même drapeau :
`ops::mesh::merge_nodes(&mesh, tol, in_place)` — miroir strict de la forme
Python, qui n'ajoute que la valeur par défaut. La brique de conteneur
sous-jacente est `SubMesh::remap_nodes(&map)`, un **renommage** de nœuds à
structure constante.
