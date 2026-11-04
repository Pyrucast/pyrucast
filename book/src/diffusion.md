# Diffusion (loi de Fick)

## En bref

La diffusion d'une espèce dans un milieu : de l'humidité dans un béton, de
l'hydrogène dans un acier, des chlorures dans un enrobage. La concentration est
l'inconnue, et l'espèce va des zones riches vers les zones pauvres.

| | |
|---|---|
| opérateur | `pyrucast.model.fick(fespace, species, symmetry="isotropic")` |
| tag | `diffusion` |
| type | domaine avec comportement (`label()` : `"Fick"`) |
| éléments | `SEG2`, `SEG3` ; `TRI3`, `QUA4`, `TRI6`, `QUA8`, `QUA9` ; `TET4`, `PYRA5`, `PENTA6`, `HEX8`, `TET10`, `PENTA15`, `HEX20`, `HEX27` ; dans des `Coords` de leur dimension, ou plus grandes (un fil, une plaque mince) |

`species` nomme l'espèce : toutes ses grandeurs portent ce nom (`c_H2`,
`j_H2`, `D_H2`), si bien que deux espèces vivent sur le même maillage sans se
confondre. Ce qui appartient au milieu ne porte pas le nom de l'espèce : le
coefficient de stockage `poro` et les axes matériau. `symmetry` choisit la
forme de la diffusivité, comme pour la [conduction](thermique.md#théorie).

L'équation est celle de la conduction, mais la physique est distincte : son tag
est `diffusion`, et non `thermal`, pour qu'un problème qui couple les deux
puisse prendre l'une sans l'autre.

## Théorie

La première loi de Fick relie le flux de l'espèce au gradient de concentration,
et la conservation de l'espèce donne, avec un coefficient de stockage
\\(\varphi\\) (la porosité pour une espèce qui diffuse dans un solide
poreux) :

\\[
\mathbf j = -\mathsf D\\,\nabla c ,
\qquad
\varphi\\,\frac{\partial c}{\partial t} - \nabla\cdot(\mathsf D\\,\nabla c) = 0 .
\\]

C'est l'équation de la chaleur avec \\(c\\) pour \\(T\\), \\(\mathsf D\\) pour
\\(\mathbf K\\) et \\(\varphi\\) pour \\(\rho\\,c_p\\). Sa forme faible et sa
discrétisation sont les mêmes :

\\[
K_{ij} = \int_e \nabla N_i^{\mathsf T}\\,\mathsf D\\,\nabla N_j\\,d\Omega ,
\qquad
C_{ij} = \int_e \varphi\\,N_i\\,N_j\\,d\Omega .
\\]

La diffusivité \\(\mathsf D\\) est un scalaire (`isotropic`) ou un tenseur
orienté par les axes matériau (`orthotropic`, `anisotropic`), construit comme
la conductivité.

## Variables

Pour l'espèce `H2` :

| | |
|---|---|
| primale | `c_H2`, la concentration |
| duale | `j_H2`, le flux d'espèce nodal |
| matériau requis | `D_H2` ; ou `D_H2_1`, `D_H2_2`, `D_H2_3` (`orthotropic`) ; ou `D_H2_11`, `D_H2_12`, `D_H2_13`, `D_H2_22`, `D_H2_23`, `D_H2_33` (`anisotropic`) ; orientés, plus les axes : `V1X`, `V1Y` en 2-D ; `V1X`, `V1Y`, `V1Z`, `V2X`, `V2Y`, `V2Z` en 3-D |
| matériau facultatif | `poro`, demandé par le stockage seul |
| entrée du comportement | `grad_c_H2_x`, `grad_c_H2_y`, `grad_c_H2_z` |
| sortie du comportement | `j_H2_x`, `j_H2_y`, `j_H2_z` |
| état | aucun |

La sortie est `𝖣·∇c`, le flux de la forme faible ; le flux de Fick est son
opposé. Elle est nommée d'après la duale de l'espèce, si bien qu'un modèle qui
porte une conduction et plusieurs espèces garde des flux distincts.

## Ce qu'elle produit

| | |
|---|---|
| raideur | \\(K\\), la diffusion |
| masse | \\(C\\), le stockage ; exige `poro` |
| géométrique | — |
| tangente | = raideur : la loi est linéaire |
| forces internes | \\(\int \nabla N_i\cdot\mathbf j\\), soit \\(K\\,c\\) |
| forces externes | aucune ; un flux entrant se donne par un chargement, ou par [`flux`](echanges.md#flux-imposé) |

## Exemple

Une barre de diffusivité 0,5 et de longueur 2, où un flux d'hydrogène de 1
entre à gauche, la concentration étant nulle à droite. Le multiplicateur de
l'appui est le flux qui sort, égal au flux qui entre :

```python
{{#include ../../tests/python/test_doc_diffusion.py:fick}}
```

Deux espèces et une conduction sur le même maillage forment un système à trois
blocs indépendants :

```python
{{#include ../../tests/python/test_doc_diffusion.py:especes}}
```

Une diffusivité qui dépend de la température se calcule par une
[évolution](evolution.md#appliquer-une-courbe-à-un-champ) appliquée au champ de
température, avant chaque assemblage. Un transfert d'espèce à travers une
interface entre deux corps est un [transfert d'interface](echanges.md#transfert-dinterface).

## Côté Rust

`Fick` (`src/models/fick.rs`) garde son sous-espace, sa `MaterialSymmetry` et
le nom de l'espèce. Ses noyaux, `element_stiffness` et `element_storage`, ont
la forme de ceux de la conduction et partagent avec eux la construction du
tenseur orienté (`symmetry::transport_tensor_from`). Les noms se construisent
depuis l'espèce : `fick::primal_var("H2")` rend `c_H2`, `fick::dual_var` rend
`j_H2`.

```rust,ignore
{{#include ../../tests/fick.rs:example}}
```
