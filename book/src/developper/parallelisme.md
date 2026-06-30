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

Toute région parallélisée soit **écrit chaque case de sortie exactement une
fois** (écriture indexée / `par_chunks_mut`), soit est une **réduction
associative sur les mêmes valeurs dans le même regroupement** (min/max). Les
résultats sont donc **bit-à-bit identiques** au séquentiel, quel que soit
`RAYON_NUM_THREADS`. L'assemblage calcule les matrices élémentaires en parallèle
puis les disperse (scatter) dans le COO **en série, en ordre de cellule** — même
garantie.

Vérification : la suite de tests (qui asservit des valeurs numériques exactes)
passe à l'identique sous `RAYON_NUM_THREADS=1` puis `=8`.

**Seule exception : le solveur** (back-end faer, pivotage/ordering différents)
n'est pas bit-à-bit identique à l'ancien LU dense, mais reste dans les tolérances
numériques.

## Ce qui reste séquentiel (et pourquoi)

- **Scatters nodaux à accumulation** — `ops::field::divergence` et
  `ops::assemble::flux` accumulent plusieurs cellules dans un même nœud partagé
  (`d[node] += …`). Paralléliser naïvement créerait une course ; une réduction
  parallèle par partiels est une évolution future, hors du périmètre « data-
  parallèle sûr ».
- **Fusion de champs** — `consolidate` / `consolidate_element` (dédup et
  vérification de cohérence entre zones).
- **Poutre de Timoshenko** — élément à deux quadratures (flexion / cisaillement),
  hors du driver à un seul espace EF ; les maillages de poutres sont 1-D et
  petits. Son intégration de comportement, elle, est mutualisée.
- **Mailleurs** — les noyaux séquentiels par nature (Bowyer–Watson, front
  avançant) restent séquentiels.

## Solveur

Le solveur utilise un **LU creux multithreadé** (faer) et met en cache la
**factorisation réutilisable** dans la `Matrix` (factor once, solve many). Détails
dans [Opérateurs de solveur](../operateurs/solveur.md). Rôles des bibliothèques :

```
nalgebra            nalgebra-sparse                faer
primitives     →    assemblage COO→CSC + serde  →  factorise & résout (parallèle, creux)
(B, J, géométrie)   stockage de la Matrice         back-end solveur uniquement
```

## Côté Python

Le parallélisme tourne **à l'intérieur** de chaque appel pyrucast (les threads
rayon ne dépendent pas du GIL). Le GIL n'est pas relâché : deux appels pyrucast
*concurrents* depuis des threads Python se sérialisent, mais un seul appel lourd
profite pleinement de tous les cœurs.
