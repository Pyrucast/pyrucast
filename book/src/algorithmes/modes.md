# Modes propres et base modale

{{#pyalgo
solver.eigen solver.quadratic_eigen matrix.project_on_basis
node_field.project_on_basis node_field.recombine}}

Quatre opérateurs travaillent avec des modes :

- `solver.eigen` cherche les modes propres de `a x = λ b x` les plus proches
  d'une valeur donnée : les vibrations avec la raideur et la masse
  (`λ = ω²`), le flambage avec la raideur et la raideur géométrique ;
- `matrix.project_on_basis` et `node_field.project_on_basis` ramènent une
  matrice et un champ de forces sur une liste de modes ;
- `node_field.recombine` fait le chemin inverse : des coordonnées modales au
  champ physique ;
- `solver.quadratic_eigen` cherche les modes complexes d'un petit problème
  amorti, typiquement projeté sur des modes réels.

## Un mode

`eigen` rend une **liste de modes**, et un mode est un `dict` (une `struct` à
membres publics, `Mode`, en Rust) :

| Clé | Contenu |
|---|---|
| `node` | le nœud propre au mode, créé à l'origine des `Coords` des matrices |
| `support` | le maillage POI1 commun aux modes de la liste, dont les nœuds sont exactement les leurs |
| `eigenvalue` | λ ; pour un mode complexe, la partie réelle de ω |
| `shape` | la déformée, un `NodeField` posé sur les supports des colonnes de la matrice, sans les multiplicateurs |
| `pairs` | les paires (primale, duale) de la déformée, sous la forme que prend `measure.xty` : `[("u_x", "f_x"), …]` |
| `generalized_mass` | φᵀBφ de la déformée, après normalisation |
| `eigenvalue_imag`, `shape_imag` | les parties imaginaires d'un mode complexe, absentes pour un mode réel |

```python
{{#include ../../../tests/python/test_doc_modes.py:solver-eigen}}
```

**Le nœud fait l'identité du mode.** Tout ce qu'on projette sur une liste de
modes est rangé au nœud de chacun : la coordonnée modale `alpha`, la force
modale `f_alpha`. Tout se lit donc par nœud — `q.value(mode["node"], "alpha")`
— et l'ordre de la liste importe peu.

**Le support est commun à la liste.** Deux champs ne se combinent que s'ils
sont posés sur le même objet support (voir
[Champ](../field.md#même-support-veut-dire-même-objet)). Toutes les matrices et
tous les champs projetés sur une même liste sont posés sur son support : ils
s'additionnent entre eux, et `solver.solve` les résout ensemble.

**Les déformées sont posées où `solve` pose ses solutions** : sur les supports
des colonnes de la matrice. Un déplacement statique et une déformée se
combinent donc.

Une base se fabrique aussi à la main, nœuds et support compris :

```python
{{#include ../../../tests/python/test_doc_modes.py:base_a_la_main}}
```

## Le calcul de `eigen`

`eigen(a, b, count, shift)` cherche les `count` valeurs propres les plus
proches du décalage `shift` (dans l'unité de λ : ω² en vibration), par ordre
croissant.

1. **La matrice décalée** `c = a − shift·b` est factorisée une fois, par la
   factorisation de `solve`, relations comprises et mises à l'échelle de la
   structure. Une structure libre a λ = 0 : avec `shift = 0`, `c` est
   singulière, et l'erreur le dit ; on lui donne un décalage négatif.
2. **L'itération** de Krylov-Schur de faer cherche les plus grandes valeurs θ
   de l'opérateur `c⁻¹ b`, qui sont `1/(λ − shift)` : les λ les plus proches du
   décalage. Elle travaille sur les seules inconnues où `b` a des termes : les
   multiplicateurs, dont l'ordre de grandeur est celui des forces de liaison,
   n'entrent pas dans ses produits scalaires. Son vecteur de départ est
   pseudo-aléatoire, tiré d'une graine fixe : le calcul se répète à
   l'identique, et aucune symétrie de ce vecteur ne cache un mode.
3. **Une résolution de plus par mode** donne le vecteur entier : les
   inconnues sans masse (des rotations sans inertie, par exemple) et les
   multiplicateurs. Les relations y sont tenues.
4. **Rayleigh-Ritz.** Les modes trouvés forment de petites matrices
   `Â = Xᵀ(a X)` et `B̂ = Xᵀ(b X)`, dont le problème propre se résout en dense :
   factorisation de Cholesky de celle des deux qui est définie positive
   (`B̂` en vibration, `Â` en flambage), puis décomposition symétrique. Deux
   modes de même valeur propre — les deux flexions d'une poutre de section
   carrée — sortent ainsi orthogonaux pour `b`.
5. **Le résidu** de chaque mode, `‖a x − λ b x‖`, rapporté à
   `‖a x‖ + |λ| ‖b x‖`, doit être sous 1e-8 ; sinon l'erreur nomme le mode.
6. **La normalisation** : `"mass"` (le défaut) ramène φᵀbφ à ±1, si bien que,
   projetée sur les modes, `b` devient l'identité et `a` la diagonale des
   valeurs propres ; `"max"` ramène à 1 la valeur de plus grande valeur
   absolue. Dans les deux cas, cette valeur est positive.

`a` et `b` doivent être symétriques, et l'une des deux définie positive sur
les inconnues que les relations laissent libres. Une relation unilatérale
(`>=`, `<=`) est refusée : elle n'a pas de problème propre.

Un `Ctrl+C` arrête le calcul entre deux étapes : après la factorisation, après
l'itération, avant la résolution de chaque mode. La factorisation et
l'itération sont chacune un seul appel à faer : un `Ctrl+C` pendant l'une
d'elles prend effet quand elle se termine.

## Projeter et recombiner

**Une matrice** : `matrix.project_on_basis(k, modes)` rend `K*ᵢⱼ = φᵢᵀ K φⱼ`,
sur le support des modes, lignes `f_alpha` et colonnes `alpha`. Chaque ligne
de `k` rencontre une déformée par la primale à laquelle sa duale est appariée :
une ligne `f_x` lit `u_x`. Les lignes des relations et les colonnes des
multiplicateurs ne sont appariées à rien : une raideur se projette de la même
façon avec ou sans ses conditions aux limites. Une matrice symétrique donne une
matrice projetée exactement symétrique.

**Un champ de forces** : `node_field.project_on_basis(f, modes)` rend
`F*ᵢ = φᵢᵀ F`, composante `f_alpha` au nœud de chaque mode. Le produit est
celui de `measure.xty` : chaque composante d'une déformée multiplie la
composante de `f` que sa paire nomme, zone par zone sur un même support. Un
champ qui ne porte aucune des duales sur les supports des déformées est refusé :
c'est un déplacement donné pour une force, ou un champ posé ailleurs, que
`node_field.restrict_like` ramène sur leurs supports.

**Recombiner** : `node_field.recombine(q, modes)` rend `u = Σ qᵢ φᵢ`, chaque
`qᵢ` lu au nœud du mode, composante `alpha`.

Le cas vibr10 de la validation enchaîne les trois, sur la base écrite à la main
plus haut : un balourd tournant excite une masse sur ressorts.

```python
{{#include ../../../tests/python/test_doc_modes.py:matrix-project_on_basis}}
{{#include ../../../tests/python/test_doc_modes.py:node_field-project_on_basis}}
{{#include ../../../tests/python/test_doc_modes.py:node_field-recombine}}
```

## Une tranche de modes

Une matrice projetée sur une liste vit sur le support de la liste, un nœud par
mode. Une tranche (`modes[1:]`) qui garderait le support complet laisserait
sans terme les nœuds des modes écartés : la matrice projetée serait singulière.
Les opérateurs vérifient donc que les nœuds du support sont exactement ceux des
modes, et refusent sinon, en donnant le remède : une tranche, ou une réunion de
deux listes, reçoit son propre support, sur les mêmes nœuds.

```python
{{#include ../../../tests/python/test_doc_modes.py:tranche}}
```

Les nœuds ne changent pas : chaque mode garde son identité, et tout ce que la
tranche projette partage son nouveau support.

## Modes complexes

`solver.quadratic_eigen(mass, stiffness, damping)` cherche les modes de
`[K + iωC − ω²M] X = 0`, avec `q(t) = X·exp(iωt)` : `Re ω` est la pulsation
amortie, `Im ω` le taux de décroissance, positif pour un mode amorti.

- Le problème est récrit au premier ordre, `z = [x ; iωx]`, et résolu en dense
  par l'algorithme QZ de faer sur des matrices `2n × 2n`. Il s'adresse donc à
  de petits problèmes : des matrices projetées sur des modes réels.
- Toutes les valeurs propres finies sortent, triées par `|ω|` puis par `Re ω` :
  un mode oscillant vient avec son miroir `−ω̄`, de déformée conjuguée.
- Les modes ont la forme de ceux de `eigen` ; `eigenvalue` et `shape` portent
  les parties réelles, `eigenvalue_imag` et `shape_imag` les parties
  imaginaires. La normalisation `"mass"` ramène `XᴴMX` à 1.
- Des matrices qui portent des relations sont refusées : leurs multiplicateurs
  n'ont pas de masse, et donneraient des valeurs propres infinies.
- Un `Ctrl+C` arrête le calcul avant ou après l'algorithme QZ, qui est un seul
  appel à faer et se termine avant que l'arrêt prenne effet.
- Calculée sur des matrices projetées, une déformée complexe porte des
  coordonnées modales : `recombine` de chacune de ses deux parties, avec les
  modes réels de la projection, la ramène aux inconnues physiques.

```python
{{#include ../../../tests/python/test_doc_modes.py:solver-quadratic_eigen}}
```

## Ce qui n'y est pas

- Une fonction pour couper ou réunir des listes de modes : la recette
  ci-dessus suffit.
- La sauvegarde d'une base : `save` ne prend pas une liste de `dict`, ni un
  `Node`.
- Les modes statiques, la projection d'un signal temporel, la recombinaison
  de vitesses, de contraintes ou de réactions.
