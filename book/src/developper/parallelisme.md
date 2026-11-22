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

- Le module `parallel` ré-exporte le prelude rayon et fixe la politique de
  grain (`MIN_PARALLEL_LEN`, via `with_min_len`) : les petits problèmes restent
  effectivement séquentiels, sans surcoût de threads.
- Les **drivers** de `models::kernel` (`element_pointwise`, `nodal_pointwise`,
  `assemble_block`, `element_block_colored`, `scatter_to_nodes`) sont les seuls
  porteurs de rayon côté physiques. Ils tiennent les guards de
  lecture, parallélisent par cellule et appellent un noyau pur fourni par la
  physique. Voir [Ajouter une physique](../ajouter-une-physique.md).

## Ce qu'un noyau au point d'intégration n'a pas le droit de faire

Un noyau au point de Gauss tourne des dizaines de millions de fois dans une
résolution non linéaire. Deux règles l'encadrent :

1. **Aucun test que l'amont a déjà tranché** — la présence d'une composante, la
   forme d'un champ, une `Option` à déballer. Les branchements qui restent sont
   ceux de la **physique** (l'essai élastique qui plastifie ou non).
2. **Aucune allocation dynamique** — ni `Vec`, ni `String`, ni `format!`, ni
   structure intermédiaire construite par point.

La vérification n'est pas supprimée, elle est **déplacée** : un champ peut avoir
été fabriqué à la main, on contrôle donc ses composantes *et leur ordre* une fois
par zone, avant la région parallèle (`Behavior::zone_layout` pour la voie point,
`Domain::element_layout` pour la voie matrice, `kernel::element_pointwise`), avec
un message qui nomme le champ et l'écart. Le
noyau reçoit alors **la ligne** de son point — une tranche empruntée du tampon —
et une table d'indices : il indexe et calcule, rien d'autre. Un noyau conforme
s'écrit sans un seul `?` sur ses lectures, ce qui rend la règle vérifiable à
l'œil.

## Zéro-copie

Les régions parallèles **n'effectuent pas de copies intermédiaires**. Plutôt que
de recopier les données dans des `Vec` avant de calculer, on **tient les
read-guards** pendant toute la région parallèle et on **emprunte les tranches
`&[f64]`** en place (`SubField::values()`, `SubMesh::connectivity()`,
`dn_dx` calculé à la volée). Les verrous de lecture sont concurrents : seuls les
écrivains attendent. L'espace éléments finis n'a aucune mutabilité intérieure
(méthodes `&self` pures), donc `&SubFiniteElementSpace` est `Sync` et appelable
depuis plusieurs threads.

**La règle ne s'arrête pas aux régions parallèles.** Une `.to_vec()` sur une
connectivité est suspecte partout, y compris dans du code séquentiel : elle coûte
4 octets par nœud et par copie, et elle se refait souvent à chaque assemblage ou à
chaque évaluation de résidu. Deux idiomes reviennent, et aucun n'oblige à recopier :

- **Un opérateur de maillage** lit toutes ses zones, puis écrit dans `Coords`
  (`add_nodes`, `decref_all`). Les deux phases sont séparées : la garde de lecture
  meurt avant l'écriture, il suffit de l'ouvrir pour la passe au lieu d'en extraire
  un tuple. Quand les deux se chevauchent vraiment, tenir la garde reste licite —
  `SubMesh` → `Coords` est l'ordre de verrouillage de tout le dépôt, jamais
  l'inverse, et `SubMesh::to_poi1` le pratique déjà.
- **Un remplisseur de bloc ou de champ** construit d'abord (un constructeur scelle,
  et sceller **écrit**), puis ouvre ses gardes et remplit. Les formes `_with`
  existent pour ça — `SubMatrix::add_entry_with`, `SubNodeField::nodes_with` : elles
  reçoivent le `&SubMesh` que l'appelant tient déjà, au lieu de reprendre un verrou
  à chaque entrée.

Les seules copies légitimes sont celles qu'on **mute** ensuite (dédoublonner, trier,
étendre) et celles qu'impose une signature : une fonction qui rend une liste sans
rendre la garde qui la porte, comme `Cell::node_ids` ou la frontière Python.

## Déterminisme

Les opérateurs de champ et d'intégration parallélisés soit **écrivent chaque
case de sortie exactement une fois** (écriture indexée / `par_chunks_mut`), soit
sont une **réduction associative sur les mêmes valeurs dans le même
regroupement** (min/max). Ces résultats sont **bit-à-bit identiques** au
séquentiel, quel que soit `RAYON_NUM_THREADS`.

L'**assemblage** suit un schéma en deux temps. Le motif creux global (sparsité
CSR) est construit depuis la seule **topologie des blocs** — indépendant des
matériaux — puis **gardé en cache par le `Model`**, donc réutilisé tel quel d'un
assemblage à l'autre (le gain majeur pour une boucle de Newton, où seuls les
matériaux changent). L'état assemblé **partage** ses tableaux d'indices avec ce
motif : d'un assemblage au suivant, seules les valeurs sont neuves.

La phase numérique **calcule et disperse dans la même passe**, cellule par
cellule, **par coloration** : deux cellules d'une même couleur ne partagent aucun
DOF, donc les cellules d'une couleur écrivent **en parallèle** dans des cases
disjointes — via un `Vec<AtomicU64>` (`Relaxed`, soit un simple `mov` sur x86,
sans `unsafe`) — les couleurs étant traitées en séquence. Chaque tâche garde
**une** matrice élémentaire de travail, réutilisée d'une cellule à l'autre :
aucun ensemble de matrices élémentaires n'est stocké en mémoire. La coloration
étant fixe, le résultat est **déterministe** (indépendant du nombre de threads),
mais **pas bit-à-bit** face à une réduction séquentielle : la somme de chaque
case est réordonnée par couleur. La dispersion **séquentielle**
`ops::scatter::scatter_serial` (en ordre de bloc, bit-à-bit) sert de référence
dans les tests ; elle calcule d'abord toutes les matrices élémentaires d'un
bloc, puis les additionne.

Le motif lui-même ne matérialise **pas** les paires `(ligne, colonne)` de chaque
entrée : elles se déduisent de la position de chaque nœud dans les supports
ligne et colonne, soit une poignée d'entiers par cellule au lieu d'une paire par
entrée. Les cases CSR, elles, sont résolues une fois et gardées à plat ; quand
l'ordre des DDL rend consécutives les colonnes d'un même nœud — ce que
`NodesThenVars` vise, et que l'assembleur **vérifie** plutôt que de le supposer —
une case de base suffit pour toutes les variables primales de ce nœud.

Le noyau élémentaire reçoit **un `CellGeom` par espace EF** du bloc : un seul
pour une physique de continuum, plusieurs — partageant un maillage, ne différant
que par la quadrature — pour un élément **multi-quadrature** (la coque épaisse de
Reissner-Mindlin : membrane et flexion en quadrature complète, cisaillement
transverse en quadrature réduite). La sparsité ne dépendant que de
la connectivité, ces éléments empruntent le **même** chemin de scatter parallèle
sans machinerie supplémentaire : seul le noyau numérique lit plusieurs géométries.

Les **dispersions aux nœuds** — `ops::node_field::divergence`,
`ops::node_field::internal_forces` (`∫ Bᵀ σ`), `ops::node_field::external_forces`
(dont la charge répartie `∫ φ N` du modèle `flux`) et
`ops::node_field::to_nodes` — procèdent toutes de la même façon : chaque
cellule calcule sa contribution locale, puis l'accumule dans ses nœuds. Elles
passent toutes par le **même driver** `kernel::scatter_to_nodes` (« intègre un
noyau élémentaire et disperse aux nœuds »), qui s'appuie sur
`parallel::colored_scatter` : **dispersion par coloration** (couleur = nœuds
disjoints, `Vec<AtomicU64>`), le tampon local tenant **par thread** — aucun
ensemble de contributions élémentaires n'est stocké, et calcul et dispersion se
font dans la **même passe parallèle** — comme l'assemblage matriciel, qui suit
le même schéma. Le driver ne connaît pas l'intégrande : l'appelant capture le
sien (champ de contrainte, densité de flux…) dans la fermeture élémentaire.
`internal_forces` *est* une divergence (de la contrainte), et
`ops::node_field::divergence` en est le cas scalaire (`n_dual = 1`) ; la charge
répartie en est l'instance pondérée par `N` plutôt que par `∇N`. Déterministe par
couleur, non bit-à-bit face à une somme en ordre de cellule.

Vérification, par des tests dédiés : `reductions_do_not_depend_on_the_thread_count`
(`containers/field.rs`) calcule les réductions sur un pool d'un thread puis de
plusieurs et exige le même résultat ; `parallel_scatter_is_deterministic`
(`ops/matrix.rs`) assemble deux fois le même modèle et exige des valeurs
identiques ; d'autres tests de `ops/matrix.rs` confrontent la dispersion
séquentielle à l'assemblage littéral.

**Non bit-à-bit, par construction :** l'assemblage parallèle et les dispersions
aux nœuds, tous déterministes par coloration, et tous dans les tolérances
numériques.

## Ce qui reste séquentiel (et pourquoi)

- **Mailleurs** — les boucles d'insertion (Bowyer–Watson, avancée du front)
  sont séquentielles par nature ; seules certaines passes sont parallèles
  (lissage du pavage, fusion de nœuds `merge_nodes`, tests d'intersection du
  mailleur volumique).
- **Dispersion d'un bloc de couplage** (`interface_transfer`) — ses matrices
  élémentaires sont calculées en parallèle, mais la dispersion est séquentielle :
  la coloration d'*une* connectivité ne garantit plus des cases disjointes
  quand lignes et colonnes vivent sur deux maillages. Une interface est un
  maillage de bord, le coût est négligeable.

## Solveur

Le solveur utilise un **LU creux multithreadé** (faer) et met en cache la
**factorisation réutilisable** dans la `Matrix` : on factorise une fois, et
chaque résolution suivante sur la même matrice la réutilise. Détails
dans [Résolution d'un système linéaire](../algorithmes/solveur.md). Rôles des bibliothèques :

```text
nalgebra            nalgebra-sparse                faer
primitives     →    stockage CSR/CSC + serde    →  factorise & résout (parallèle, creux)
(B, J, géométrie)   Matrice (scatter → CSR)        back-end solveur uniquement
```

## Côté Python

Le parallélisme tourne **à l'intérieur** de chaque appel pyrucast (les threads
rayon ne dépendent pas du GIL). Le GIL n'est relâché que par
`pyrucast.export.to_arrays` : deux autres appels pyrucast *concurrents* depuis
des threads Python se sérialisent, mais un seul appel lourd profite pleinement
de tous les cœurs.
