# Barre / treillis

Élément `SEG2` à 2 nœuds transmettant uniquement l'**effort axial** (treillis).

## Équations résolues

La barre suit la loi axiale 1-D `N = E·A·ε`, avec la déformation `ε = du/ds`
(dérivée du déplacement *le long* de la barre). L'orientation est déduite des
coordonnées des nœuds via le cosinus directeur `c = (x_B − x_A)/L`, si bien que
la rigidité élémentaire **globale** s'écrit, en `d` dimensions,

\\[
K_e = \frac{E\,A}{L}
\begin{bmatrix} c\,c^\top & -c\,c^\top \\\\ -c\,c^\top & c\,c^\top \end{bmatrix},
\\]

écrite aux positions `(NodeId_i, f_a) × (NodeId_j, u_b)`. Le même code fonctionne
en 1-D, 2-D et 3-D.

- **primal** : `u_x, u_y(, u_z)` — **dual** : `f_x, f_y(, f_z)`.
- **matériau** : `E` (module d'Young), `A` (section).
- **comportement** (`COMP`) : effort axial `N = E·A·(cᵀ ε c)`, à partir de la
  déformation `ε` (op [`deformation`](../node-field.md)).

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
