# Parallélisme

pyrucast parallélise ses boucles de calcul lourdes (assemblage par élément,
intégration du comportement par point de Gauss, mathématiques de champ, export
VTK, solveur) sur les cœurs CPU avec [rayon](https://docs.rs/rayon). Le
parallélisme est **toujours actif** ; le nombre de threads suit
`RAYON_NUM_THREADS`.

## Le parallélisme est porté *au-dessus* des noyaux

Principe central : **un noyau de physique ne voit jamais rayon, le store, ni un
verrou.** La parallélisation vit dans une couche au-dessus, de sorte qu'ajouter
une physique ou un opérateur revient à écrire des mathématiques **séquentielles
pures**.

- Le module [`parallel`](#) ré-exporte le prelude rayon et fixe la politique de
  grain (`MIN_PARALLEL_LEN`, via `with_min_len`) : les petits problèmes restent
  effectivement séquentiels, sans surcoût de threads.
- Les **drivers** de `models::kernel` (`integrate_pointwise`, `assemble_block`)
  sont les seuls porteurs de rayon côté physiques. Ils tiennent les guards de
  lecture, parallélisent par cellule et appellent un noyau pur fourni par la
  physique. Voir [Ajouter une physique](../ajouter-une-physique.md).

## Zéro-copie

Les régions parallèles **n'effectuent pas de copies intermédiaires**. Plutôt que
de recopier les données du store dans des `Vec` avant de calculer, on **tient les
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
matériaux changent). Les matrices élémentaires sont calculées en parallèle, puis
**dispersées dans le CSR par coloration des cellules** : deux cellules d'une même
couleur ne partagent aucun DOF, donc les cellules d'une couleur écrivent **en
parallèle** dans des cases disjointes — via un `Vec<AtomicU64>` (`Relaxed`, soit
un simple `mov` sur x86, sans `unsafe`) — les couleurs étant traitées en
séquence. La coloration étant fixe, le résultat est **déterministe** (indépendant
du nombre de threads), mais **pas bit-à-bit** face à une réduction séquentielle :
la somme de chaque case est réordonnée par couleur. Un chemin de scatter
**séquentiel** (en ordre de bloc), lui bit-à-bit, est conservé comme référence de
test.

Le noyau élémentaire reçoit **un `CellGeom` par espace EF** du bloc : un seul
pour une physique de continuum, plusieurs — partageant un maillage, ne différant
que par la quadrature — pour un élément **multi-quadrature** (poutre de
Timoshenko flexion/cisaillement, coque à venir). La sparsité ne dépendant que de
la connectivité, ces éléments empruntent le **même** chemin de scatter parallèle
sans machinerie supplémentaire : seul le noyau numérique lit plusieurs géométries.

Les **scatters nodaux** — les opérateurs `Bᵀ` (`ops::field::divergence` et les
forces internes `ops::internal_forces`, Cast3m `BSIG`) et la charge répartie
`ops::assemble::flux` (`∫ φ N`) — dispersent tous de la même façon : chaque
cellule calcule sa contribution locale, puis l'accumule dans ses nœuds. Ils
partagent le helper `parallel::colored_scatter` : **même scatter par coloration**
(couleur = nœuds disjoints, `Vec<AtomicU64>`), le tampon local tenant **par
thread** — aucun ensemble élémentaire n'est matérialisé, et calcul et scatter se
font dans la **même passe parallèle** (contrairement à l'assemblage, qui calcule
d'abord les matrices élémentaires). `internal_forces` *est* une divergence (de la
contrainte) : les deux passent par le même noyau `kernel::divergence`, dont
`ops::field::divergence` n'est que le cas scalaire (`n_dual = 1`). Déterministe
par couleur, non bit-à-bit face à une somme en ordre de cellule.

Vérification : la suite de tests passe sous `RAYON_NUM_THREADS=1` puis `=8` ; les
opérateurs write-once / réduction sont asservis à des valeurs exactes,
l'assemblage parallèle à l'assemblage littéral **à tolérance**, plus un test de
déterminisme.

**Non bit-à-bit, par construction :** l'assemblage parallèle et les scatters
nodaux (`divergence`, forces internes, `flux`) — tous déterministes par
coloration — et **le solveur** (back-end faer, pivotage/ordering différents de
l'ancien LU dense). Tous dans les tolérances numériques.

## Ce qui reste séquentiel (et pourquoi)

- **Fusion de champs** — `consolidate` / `consolidate_element` (dédup et
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
