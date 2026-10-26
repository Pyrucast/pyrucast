# Opérateurs d'assemblage

Le module `ops::matrix` transforme un [`Model`](../model.md) en une
[`Matrice`](../matrix.md) (raideur, masse). Les **seconds membres** répartis
— `internal_forces`, `external_forces` — sont eux aussi des assemblages, mais leur
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

## `tangent(model, materials, deformation, prev=None, dt=None)` → `Matrix`

Assemble la **matrice tangente cohérente (algorithmique)** `K_t = ∫ Bᵀ D_alg B`
(Cast3M `KTAN`), qui donne la convergence quadratique du Newton non-linéaire.

Il prend **les mêmes arguments que `behavior.integrate`**, et pour la même
raison : `D_alg` est la dérivée du *pas* `ε(B) ↦ σ(B)` à état A figé, donc les
deux extrémités du pas sont dans sa définition. `prev=None` vaut l'état de repos.

Aucun champ de modules n'est matérialisé : il n'aurait eu que cet assembleur pour
lecteur, et il pesait de 6 à 21 réels par point de Gauss. `D_alg` est évalué au
point, ce qui a aussi sorti sa dérivation de `behavior.integrate` — pour les huit
lois plastiques sans forme fermée, COMP payait treize retours radiaux par point
au lieu d'un, à chaque itération, que la tangente serve ou non.
`model.tangent_source()` dit d'avance ce qu'une tangente coûtera.
`materials` résout chaque zone comme `stiffness`.

Une physique **linéaire** (conduction, Fick, échanges, barres, poutres, coques)
et les **contraintes** (Dirichlet, MPC, contact…) n'ont pas de loi à dériver :
leur tangente est leur raideur, et c'est elle qui est assemblée, sans lire
`deformation`. Une physique non linéaire sans tangente propre (l'endommagement,
pour l'instant) fait échouer l'appel plutôt que de rendre un bloc nul.

```python
{{#include ../../../tests/python/test_doc_ops_assemblage.py:tangent}}
```

## Composition : `assemble(&mut Matrix)`

Une matrice est assemblée **une fois**, sur la configuration active à ce
moment-là. `stiffness`, `mass`, `geometric` et `tangent` rendent une matrice
assemblée, dont les valeurs ne bougent plus : ni quand une autre configuration
devient active, ni par un second `assemble()`, qui ne fait alors rien. Les
valeurs d'une autre configuration s'obtiennent en refaisant la matrice.

`assemble()` sert aux blocs réunis **sans leurs valeurs** : une `SubMatrix`
ajoutée par `add_sub`, l'union `|`, `filter`, `exclude`. Les blocs **calculés**
de `stiffness` (une recette, des valeurs produites au scatter), `Matrix::finalize`
ne sait pas les assembler seul ; `m.assemble()` le fait, sur la configuration
active. C'est une **méthode** et non une fonction libre : elle mute un seul
conteneur en préservant son invariant, exactement comme sa voisine `finalize`.
Elle construit le motif creux depuis les **blocs seuls** (sans `Model`) et y
disperse les valeurs :

```rust,ignore
{{#include ../../../tests/doc_conteneurs.rs:reassembler}}
```

```python
{{#include ../../../tests/python/test_doc_ops_assemblage.py:assemble}}
```

Contrairement à `stiffness`, ce chemin ne consulte pas le motif mémoïsé sur le
`Model` (il n'y a pas de `Model` ici) et reconstruit la sparsité à chaque appel —
adapté à la composition ponctuelle ; l'assemblage répété d'un modèle fixe
reste sur `stiffness`.

## La somme

`+` additionne les **valeurs** de ses opérandes, chacune telle qu'elle a été
assemblée, sur l'union de leurs DOFs : le résultat est assemblé et se résout
tel quel. Rien n'est recalculé. Avec le **facteur scalaire** des blocs
(`bloc * s`, `bloc / s` — voir
[Matrice creuse](../matrix.md#facteur-scalaire-et-somme-de-matrices)), c'est
l'opérateur de la dynamique, `M/dt + K` :

```python
{{#include ../../../tests/python/test_doc_ops_assemblage.py:somme}}
```

Deux matrices assemblées sur deux configurations s'additionnent donc telles
quelles. C'est ce que fait le pas à pas en grands déplacements : la capacité
sur la référence, la conduction et les échanges sur la pièce déformée (voir
[Grandes transformations](../mecanique/grandes-transformations.md)).

```python
{{#include ../../../tests/python/test_doc_ops_assemblage.py:somme_configurations}}
```

Un opérande dont des blocs calculés n'ont jamais été assemblés (une sortie de
`filter` ou de `|`) n'a pas de valeurs à additionner : la somme lève une erreur
plutôt que de les calculer en silence sur la configuration du moment. `get` fait
de même.

## Chargement réparti : `model.flux(fespace, target, dual)` → `Model`

L'analogue de `FLUX` / `SOUR` de cast3m. Ce n'est **pas** un opérateur mais une
**physique** : une charge répartie est un terme de la forme variationnelle comme
un autre, simplement le premier dont le terme entier siège à droite du signe
égal. Sa dérivée par rapport à la solution est nulle, donc elle ne contribue à
aucune matrice ; on lui demande sa contribution par
[`external_forces`](comportement.md).

Elle transforme une densité `φ` — lue dans le matériau sous le nom
`phi_<dual>` — répartie sur un bord (ou un volume) en **charges nodales
cohérentes**

\\[
f_i = \int_\Gamma \varphi\\, N_i\\, d\Gamma
\approx \sum_{\text{cell}} \sum_g \varphi(\text{cell}, g)\\, N_i(\xi_g)\\, |J|_g\\, w_g,
\\]

accumulées par nœud dans un `NodeField` — une zone par sous-espace EF — sur la
ligne **duale** `dual` (par exemple `"q"` en thermique, `"f_x"` en mécanique).
Une charge n'a pas de primale : elle écrit dans la ligne duale d'une autre
physique et n'introduit aucune inconnue.

C'est aussi pourquoi elle reçoit `target`, le **modèle qu'elle charge**. Le nom
de la ligne duale ne dit pas à quelle nature elle appartient — l'utilisateur le
choisit librement —, mais le modèle chargé, si : on y cherche le sous-modèle qui
déclare cette duale, ce qui donne du même coup la nature de la charge et la
preuve que la ligne est bien assemblée par quelqu'un. Une duale mal tapée bâtit
alors une erreur de construction, là où elle produisait une charge muette.


La densité vaut ce que le champ matériau y met : uniforme si on la passe en
scalaire à `material_field`, variable par point de Gauss si on bâtit
l'`ElementField` soi-même. Un ambiant oublié n'est plus possible non plus — une
densité absente est refusée à l'assemblage, par son nom.

La mesure `|J|` venant du sous-espace EF, un **bord** s'intègre directement :
une arête `SEG2` plongée dans un `Coords` 2-D s'intègre comme une **ligne**
(Jacobien manifold), une surface comme une aire.

**Le mode normal : une pression.** La ligne `"n"` désigne une densité portée par
la **normale** de la maille, versée dans toutes les lignes de déplacement de la
cible à la fois (`f_x`, `f_y`, `f_z`) — le `PRES` de Cast3M :

\[
f_{ia} = \int_\Gamma \varphi_n\, n_a\, N_i\, d\Gamma .
\]

La normale est celle de la configuration **active** : `(t_y, −t_x)` pour une
arête de tangente `t`, qui sort de la matière quand le contour est parcouru dans
le sens trigonométrique ; une pression `p` qui pousse sur la paroi s'écrit alors
`phi_n = −p`. Évaluée sur une configuration posée en `X + u`, la charge **suit
la paroi** — sans raideur de chargement, comme dans Cast3M. Voir
[Grandes transformations](../mecanique/grandes-transformations.md).

Rien n'oblige une charge à rejoindre le modèle qu'elle charge : ne contribuant à
aucune matrice, elle se tient très bien en modèle à elle seule, avec sa propre
densité. C'est ce qu'on fait quand deux charges alimentent la même ligne duale
avec des densités différentes.

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
