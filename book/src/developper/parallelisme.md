# Parallélisme

pyrucast parallélise ses boucles de calcul lourdes (assemblage par élément,
intégration du comportement par point de Gauss, mathématiques de champ, export
VTK, solveur) sur les cœurs CPU avec [rayon](https://docs.rs/rayon). Le
parallélisme est **toujours actif** ; le nombre de threads suit
`RAYON_NUM_THREADS`.

## Le parallélisme est porté *au-dessus* des noyaux

Principe central : **un noyau de physique ne voit jamais rayon, un handle, ni un
verrou.** La parallélisation vit dans une couche au-dessus, de sorte qu'ajouter
une physique ou un opérateur revient à écrire des mathématiques **séquentielles
pures**.

- Le module [`parallel`](#) ré-exporte le prelude rayon et fixe la politique de
  grain (`MIN_PARALLEL_LEN`, via `with_min_len`) : les petits problèmes restent
  effectivement séquentiels, sans surcoût de threads.
- Les **drivers** de `models::kernel` (`element_pointwise`, `nodal_pointwise`,
  `assemble_block`)
  sont les seuls porteurs de rayon côté physiques. Ils tiennent les guards de
  lecture, parallélisent par cellule et appellent un noyau pur fourni par la
  physique. Voir [Ajouter une physique](../ajouter-une-physique.md).

## Ce qu'un noyau au point d'intégration n'a pas le droit de faire

Un noyau au point de Gauss tourne des dizaines de millions de fois dans une
résolution non linéaire. Deux invariants l'encadrent :

1. **Aucun test que l'amont a déjà tranché** — la présence d'une composante, la
   forme d'un champ, une `Option` à déballer. Les branchements qui restent sont
   ceux de la **physique** (l'essai élastique qui plastifie ou non).
2. **Aucune allocation dynamique** — ni `Vec`, ni `String`, ni `format!`, ni
   structure intermédiaire construite par point.

La vérification n'est pas supprimée, elle est **déplacée** : un champ peut avoir
été fabriqué à la main, on contrôle donc ses composantes *et leur ordre* une fois
par zone, avant la région parallèle (`Domain::zone_layout` pour la voie point,
`Domain::element_layout` pour la voie matrice, `kernel::element_pointwise`), avec
un message qui nomme le champ et l'écart. Le
noyau reçoit alors **la ligne** de son point — une tranche empruntée du tampon —
et une table d'indices : il indexe et calcule, rien d'autre. Un noyau conforme
s'écrit sans un seul `?` sur ses lectures, ce qui rend la règle vérifiable à
l'œil.

Ce que cela vaut : sur la poutre console élasto-plastique 400×80, ~55 % du temps
CPU partait dans la résolution de noms, le `format!`, le hachage et l'allocateur.

## Zéro-copie

Les régions parallèles **n'effectuent pas de copies intermédiaires**. Plutôt que
de recopier les données dans des `Vec` avant de calculer, on **tient les
read-guards** pendant toute la région parallèle et on **emprunte les tranches
`&[f64]`** en place (`SubField::values()`, `SubMesh::connectivity()`,
`dn_dx` calculé à la volée). Les verrous de lecture sont concurrents : seuls les
écrivains attendent. L'espace éléments finis n'a aucune mutabilité intérieure
(méthodes `&self` pures), donc `&SubFiniteElementSpace` est `Sync` et appelable
depuis plusieurs threads.

## Déterminisme

Les opérateurs de champ et d'intégration parallélisés soit **écrivent chaque
case de sortie exactement une fois** (écriture indexée / `par_chunks_mut`), soit
sont une **réduction associative sur les mêmes valeurs dans le même
regroupement** (min/max). Ces résultats sont **bit-à-bit identiques** au
séquentiel, quel que soit `RAYON_NUM_THREADS`.

L'**assemblage** suit un schéma en deux temps. Le motif creux global (sparsité
CSR) est construit depuis la seule **topologie des blocs** — indépendant des
matériaux — puis **mémoïsé sur le `Model`**, donc réutilisé tel quel d'un
assemblage à l'autre (le gain majeur pour une boucle de Newton, où seuls les
matériaux changent). L'état assemblé **partage** ses tableaux d'indices avec ce
motif : d'un assemblage au suivant, seules les valeurs sont neuves.

La phase numérique **calcule et disperse dans la même passe**, cellule par
cellule, **par coloration** : deux cellules d'une même couleur ne partagent aucun
DOF, donc les cellules d'une couleur écrivent **en parallèle** dans des cases
disjointes — via un `Vec<AtomicU64>` (`Relaxed`, soit un simple `mov` sur x86,
sans `unsafe`) — les couleurs étant traitées en séquence. Chaque tâche garde
**une** matrice élémentaire de travail, réutilisée d'une cellule à l'autre :
aucun ensemble élémentaire n'est matérialisé, là où les matérialiser toutes
coûtait, sur un solide, des dizaines de gigaoctets écrits puis relus une fois.
La coloration étant fixe, le résultat est **déterministe** (indépendant du nombre
de threads), mais **pas bit-à-bit** face à une réduction séquentielle : la somme
de chaque case est réordonnée par couleur. Un chemin de scatter **séquentiel**
(en ordre de bloc), lui bit-à-bit, est conservé comme référence de test — c'est
le seul qui calcule encore toutes les matrices élémentaires d'abord.

Le motif lui-même ne matérialise **pas** les paires `(ligne, colonne)` de chaque
entrée : elles se déduisent de la position de chaque nœud dans les supports
ligne et colonne, soit une poignée d'entiers par cellule au lieu d'une paire par
entrée. Les cases CSR, elles, sont résolues une fois et gardées à plat ; quand
l'ordre des DDL rend consécutives les colonnes d'un même nœud — ce que
`NodesThenVars` vise, et que l'assembleur **vérifie** plutôt que de le supposer —
une case de base suffit pour toutes les variables primales de ce nœud.

Le noyau élémentaire reçoit **un `CellGeom` par espace EF** du bloc : un seul
pour une physique de continuum, plusieurs — partageant un maillage, ne différant
que par la quadrature — pour un élément **multi-quadrature** (poutre de
Timoshenko flexion/cisaillement, coque à venir). La sparsité ne dépendant que de
la connectivité, ces éléments empruntent le **même** chemin de scatter parallèle
sans machinerie supplémentaire : seul le noyau numérique lit plusieurs géométries.

Les **scatters nodaux** — les opérateurs `Bᵀ` (`ops::node_field::divergence` et les
forces internes `ops::node_field::internal_forces`, Cast3m `BSIG`) et la charge répartie
`ops::node_field::flux` (`∫ φ N`) — dispersent tous de la même façon : chaque
cellule calcule sa contribution locale, puis l'accumule dans ses nœuds. Ils
passent tous par le **même driver** `kernel::scatter_to_nodes` (« intègre un
noyau élémentaire et disperse aux nœuds »), qui s'appuie sur le helper
`parallel::colored_scatter` : **scatter par coloration** (couleur = nœuds
disjoints, `Vec<AtomicU64>`), le tampon local tenant **par thread** — aucun
ensemble élémentaire n'est matérialisé, et calcul et scatter se font dans la
**même passe parallèle** — comme l'assemblage matriciel, qui suit le même
schéma. Le driver est **agnostique à l'intégrande** : l'appelant
capture le sien (champ de contrainte, densité de flux…) dans la closure
élémentaire. `internal_forces` *est* une divergence (de la contrainte), et
`ops::node_field::divergence` en est le cas scalaire (`n_dual = 1`) ; `flux` en est
l'instance « masse » pondérée par `N` plutôt que par `∇N`. Déterministe par
couleur, non bit-à-bit face à une somme en ordre de cellule.

Vérification : la suite de tests passe sous `RAYON_NUM_THREADS=1` puis `=8` ; les
opérateurs write-once / réduction sont asservis à des valeurs exactes,
l'assemblage parallèle à l'assemblage littéral **à tolérance**, plus un test de
déterminisme.

**Non bit-à-bit, par construction :** l'assemblage parallèle et les scatters
nodaux (`divergence`, forces internes, `flux`) — tous déterministes par
coloration — et **le solveur** (back-end faer, pivotage/ordering différents de
l'ancien LU dense). Tous dans les tolérances numériques.

## Ce qui reste séquentiel (et pourquoi)

- **Fusion de champs** — `node_field.consolidate` / `element_field.consolidate` (dédup et
  vérification de cohérence entre zones).
- **Mailleurs** — les noyaux séquentiels par nature (Bowyer–Watson, front
  avançant) restent séquentiels.

## Solveur

Le solveur utilise un **LU creux multithreadé** (faer) et met en cache la
**factorisation réutilisable** dans la `Matrix` (factor once, solve many). Détails
dans [Opérateurs de solveur](../operateurs/solveur.md). Rôles des bibliothèques :

```text
nalgebra            nalgebra-sparse                faer
primitives     →    stockage CSR/CSC + serde    →  factorise & résout (parallèle, creux)
(B, J, géométrie)   Matrice (scatter → CSR)        back-end solveur uniquement
```

## Côté Python

Le parallélisme tourne **à l'intérieur** de chaque appel pyrucast (les threads
rayon ne dépendent pas du GIL). Le GIL n'est pas relâché : deux appels pyrucast
*concurrents* depuis des threads Python se sérialisent, mais un seul appel lourd
profite pleinement de tous les cœurs.
