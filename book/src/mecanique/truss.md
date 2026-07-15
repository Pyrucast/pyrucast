# Barre / treillis

Élément `SEG2` à 2 nœuds transmettant uniquement l'**effort axial** (treillis).
Fonctionne à l'identique en 1-D, 2-D et 3-D.

## Équations continues résolues

La barre suit la loi axiale 1-D et son équilibre le long de l'axe `s` :

\\[
N = E\,A\,\varepsilon, \qquad
\varepsilon = \frac{du}{ds}, \qquad
\frac{dN}{ds} + f = 0,
\\]

où `N` est l'effort normal, `ε` la déformation axiale (dérivée du déplacement
*le long* de la barre) et `f` la charge axiale répartie. L'orientation est
déduite des coordonnées des nœuds via le cosinus directeur `c = (x_B − x_A)/L`.

## Forme discrétisée

Avec l'interpolation linéaire `SEG2`, la déformation est **constante** par
élément. Projetée sur les directions physiques par `c`, la rigidité élémentaire
**globale** (en `d` dimensions) s'écrit

\\[
K_e = \frac{E\,A}{L}
\begin{bmatrix} c\,c^\top & -c\,c^\top \\\\ -c\,c^\top & c\,c^\top \end{bmatrix},
\\]

écrite aux positions `(NodeId_i, f_a) × (NodeId_j, u_b)`. En 1-D, `c = 1` et l'on
retrouve `(EA/L)[[1,−1],[−1,1]]`.

## Variables et matériau

- **primal** : `u_x, u_y(, u_z)` — **dual** : `f_x, f_y(, f_z)`.
- **matériau** : `E` (module d'Young), `A` (section) ; `rho` **facultatif** (masse).
- **comportement** (`COMP`) : effort axial `N = E·A·(cᵀ ε c)`, à partir de la
  déformation `ε` (op [`deformation`](../operateurs/champs.md)).

> ⚠️ Une barre n'a **aucune raideur transversale** : pour un système bien posé
> il faut bloquer les DOFs transverses (treillis triangulé, appuis), sinon la
> matrice est singulière.

## Mise en donnée (Rust, testé)

Barre horizontale encastrée à gauche, appuyée transversalement à droite, force
axiale `F` au bout ⇒ `u_x = F·L/(E·A)`. Le code est le test d'intégration
`tests/truss.rs`, exécuté à chaque `cargo test` :

```rust,ignore
{{#include ../../../tests/truss.rs:example}}
```

## Exemple Python

```python
{{#include ../../../examples/truss.py}}
```

## Compléments

### Masse & rigidité géométrique

En plus de la rigidité, la barre assemble :

- **masse consistante** `M = (ρAL/6)[[2,1],[1,2]]` sur chaque composante de
  translation (`rho` composante matériau optionnelle) — `pyrucast.assemble.mass` ;
- **rigidité géométrique** `K_g = (N/L)·(I − c⊗c)` transverse, sous l'effort
  axial `N` (sortie `n` du comportement) — `pyrucast.assemble.geometric`.
