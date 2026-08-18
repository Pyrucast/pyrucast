# Opérateurs d'assemblage

Le module `ops::matrix` transforme un [`Model`](../model.md) en une
[`Matrice`](../matrix.md) (raideur, masse). Les **seconds membres** répartis
— `flux`, `internal_forces` — sont eux aussi des assemblages, mais leur
résultat est un vecteur nodal : on se range par la sortie, ils vivent donc
sous `ops::node_field`. Les intégrandes par physique vivent sous `src/models/` ; cette couche
**oriente** : boucle sur les sous-modèles, mise en place des DOFs, accumulation
dans la matrice globale.

## `stiffness(model, materials)` → `Matrix`

Assemble la **matrice de raideur** `K` couvrant **tous** les DOFs du modèle
(primaux ⊕ multiplicateurs). Chaque [`SubModel`](../model.md) contribue un ou
plusieurs blocs `SubMatrix` (une physique volumique → 1 bloc ; une contrainte
de Dirichlet → les blocs `C` + `Cᵀ`), accumulés dans **une seule** matrice. Les
conditions limites n'ont pas de statut spécial : ce sont des sous-modèles comme
les autres.

`materials` est l'[`ElementField`](construction.md) des propriétés : pour chaque
sous-modèle qui en a besoin, l'assembleur sélectionne la zone dont le
`SubFiniteElementSpace` correspond au sien (matériaux par zone). Les
sous-modèles sans matériau (Dirichlet…) ignorent ce champ.

```python
{{#include ../../../tests/python/test_doc_ops_assemblage.py:stiffness}}
```

## `mass(model, materials)` → `Matrix`

Assemble la **matrice de masse consistante** `M` (Cast3M `MASS`), ou la
**matrice de capacité thermique** `C` pour un modèle thermique (Cast3M `CAPA`).
La mécanique assemble `M = ∫ ρ · N_i N_j dx` (matériau `rho`) ; la conduction
assemble `C = ∫ ρ c_p · N_i N_j dx` (matériau `rho`, `cp`). Une physique sans
terme de masse (bord de convection, contrainte de Lagrange) ne contribue rien.

`materials` fournit les coefficients par zone, exactement comme `stiffness`. La
densité `rho` est une composante **facultative** des physiques mécaniques (comme
`alpha`), `rho` et `cp` des physiques thermiques : la raideur / conductivité
n'en a pas besoin, mais la masse / capacité les exige (erreur claire sinon).

```python
{{#include ../../../tests/python/test_doc_ops_assemblage.py:mass}}
```

## `lump(matrix)` → `Matrix`

**Concentre** (lumping, Cast3M `LUMP`) une matrice assemblée en une matrice
**diagonale** par **somme de lignes** : chaque terme diagonal devient la somme de
sa ligne, les extra-diagonaux sont supprimés. Appliqué à une matrice de masse /
capacité consistante, on obtient la masse **diagonale (lumpée)**, qui conserve la
masse totale (`Σ_i M_lump[i,i] = Σ_ij M[i,j]`) — la forme découplée bon marché des
schémas explicites. La matrice d'entrée doit être assemblée et carrée.

```python
{{#include ../../../tests/python/test_doc_ops_assemblage.py:lump}}
```

## `geometric(model, materials, stress)` → `Matrix`

Assemble la **matrice de rigidité géométrique** (initial-stress) `K_g`
(Cast3M `KSIG`) : `K_g = ∫ Gᵀ σ̂ G`, le terme de raidissement sous précontrainte,
pour le flambement et les analyses précontraintes. Le noyau
`K_g[(i,a),(j,b)] = δ_ab ∫ ∇N_i · σ · ∇N_j` est indépendant de la loi.

`stress` est le champ de contrainte de Cauchy courant (composantes Voigt
`sigma_*`, typiquement la sortie de `behavior.integrate`), résolu par zone comme
`materials`. `materials` sert encore à résoudre chaque zone mécanique (`E`, `nu`).

```python
{{#include ../../../tests/python/test_doc_ops_assemblage.py:geometric}}
```

## `tangent(model, materials, state)` → `Matrix`

Assemble la **matrice tangente cohérente (algorithmique)** `K_t = ∫ Bᵀ D_alg B`
(Cast3M `KTAN`), qui donne la convergence quadratique du Newton non-linéaire.

`state` est le champ de comportement produit par `behavior.integrate` à l'itéré
courant : en plus de la contrainte, il porte le module algorithmique `D_alg` par
point de Gauss (les composantes `ktan_*`), que cet assembleur relit. Pour une
physique **linéaire** (élasticité) la tangente vaut la rigidité élastique et
`state` est ignoré. Le module cohérent est le **producteur/consommateur** :
`integrate_point` (plasticité J2, Mazars) émet `D_alg`, cet opérateur le
consomme. `materials` résout chaque zone comme `stiffness`.

```python
{{#include ../../../tests/python/test_doc_ops_assemblage.py:tangent}}
```

## Composition : `assemble(&mut Matrix)`

`stiffness` produit une matrice portant des blocs **calculés** (recette, valeurs
produites au scatter) que `Matrix::finalize` ne sait pas assembler seul. Pour
**recomposer** — ajouter une `SubMatrix` de provenance quelconque à une matrice
existante (ou combiner plusieurs `Matrix` déjà assemblées via l'union `|`) puis
réassembler — `m.assemble()`. C'est une **méthode** et non une fonction libre :
elle mute un seul conteneur en préservant son invariant, exactement comme sa
voisine `finalize`. Elle reconstruit le motif creux depuis les **blocs seuls**
(sans `Model`) et redisperse les valeurs :

```rust,ignore
let mut k = matrix::stiffness(&model, &materials)?;
k.add_sub(Handle::new(bloc_supplementaire))?;   // invalide l'état assemblé
k.assemble()?;                             // réassemble, nouveau bloc inclus
```

```python
{{#include ../../../tests/python/test_doc_ops_assemblage.py:assemble}}
```

Contrairement à `stiffness`, ce chemin ne consulte pas le motif mémoïsé sur le
`Model` (il n'y a pas de `Model` ici) et reconstruit la sparsité à chaque appel —
adapté à la composition ponctuelle ; le réassemblage à chaud d'un modèle fixe
reste sur `stiffness`.

C'est aussi le chemin de composition pour la dynamique : chaque `SubMatrix`
porte un **facteur scalaire** paresseux (`bloc * s` / `bloc / s`, `1.0` par
défaut — voir [Matrice creuse](../matrix.md#facteur-scalaire-mulf64--divf64-et-combinaison-de-matrices)),
et l'assembleur somme déjà les contributions d'un même DOF, donc `M/dt + K`
s'obtient sans opérateur dédié :

```python
{{#include ../../../tests/python/test_doc_ops_assemblage.py:somme}}
```

## Chargement réparti : `flux(fespace, density, component)` → `NodeField`

L'analogue de `FLUX` / `PRES` de cast3m : transforme une densité de flux `φ`
répartie sur un bord (ou un volume) en **charges nodales cohérentes**

\\[
f_i = \int_\Gamma \varphi\\, N_i\\, d\Gamma
\approx \sum_{\text{cell}} \sum_g \varphi(\text{cell}, g)\\, N_i(\xi_g)\\, |J|_g\\, w_g,
\\]

accumulées par nœud dans un `NodeField` sur la composante **duale** `component`
(par exemple `"q"` en thermique, `"f_x"` en mécanique). `density` est soit une
**constante** (flux uniforme), soit un `SubElementField` (densité par élément).

La mesure `|J|` venant du sous-espace EF, un **bord** s'intègre directement :
une arête `SEG2` plongée dans un `Coords` 2-D s'intègre comme une **ligne**
(Jacobien manifold), une surface comme une aire. On compose ensuite le résultat
avec le reste du chargement par l'union `|`.

```python
{{#include ../../../tests/python/test_doc_ops_assemblage.py:flux}}
```

Exemples complets de bout en bout : [Conduction thermique](../thermique.md)
(carré chauffé) et [Élasticité](../mecanique/elasticite.md) (traction).

## Règle invariante : un Model = une Matrice

`stiffness` et `mass` produisent chacune **une seule** `Matrix` pour tout le
modèle. Le solveur reçoit donc une matrice + un second membre — pas de système
point-selle composé à jongler côté utilisateur. Voir
[Modèle physique](../model.md) et [Solveur](solveur.md).
