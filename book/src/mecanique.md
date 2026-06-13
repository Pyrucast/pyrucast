# Mécanique

Les physiques mécaniques implémentées. Comme pour la thermique, ce sont des
variantes de [`SubModel`](model.md) : chacune déclare ses variables, son
matériau, et assemble sa rigidité `K` (et son comportement `COMP`).

Convention de nommage : **primal** = déplacements `u_x, u_y, u_z` (les
inconnues) ; **dual** = forces nodales `f_x, f_y, f_z` (second membre /
réactions).

## Barre / treillis (`models/truss.rs`)

Élément `SEG2` à **2 nœuds** ne transmettant que l'**effort axial**.
L'orientation est lue dans les coordonnées des nœuds (cosinus directeurs
`c = (x_B − x_A)/L`), donc le même code marche en 1-D, 2-D et 3-D. La rigidité
élémentaire globale est

\\[
K_e = \frac{E\,A}{L}
\begin{bmatrix} c\,c^\top & -c\,c^\top \\\\ -c\,c^\top & c\,c^\top \end{bmatrix},
\\]

écrite aux positions `(NodeId_i, f_a) × (NodeId_j, u_b)`.

- **primal** : `u_x, u_y(, u_z)` — **dual** : `f_x, f_y(, f_z)`.
- **matériau** : `E` (module d'Young), `A` (section).
- **comportement** (`COMP`) : effort axial `N = E·A·(cᵀ ε c)` à partir de la
  déformation `ε` (op [`deformation`](node-field.md)).

> ⚠️ Une barre n'a **aucune raideur transversale** : pour un système bien posé,
> il faut bloquer les DOFs transverses (treillis triangulé, ou appuis adéquats),
> sinon la matrice est singulière.

### Exemple : barre en traction

Barre horizontale de longueur `L`, encastrée à gauche (`u_x = u_y = 0`),
appuyée transversalement à droite (`u_y = 0`), force axiale `F` à droite.
Solution analytique : `u_x = F·L / (E·A)`. L'exemple ci-dessous est le test
d'intégration `tests/truss.rs` (exécuté à chaque `cargo test`) :

```rust,ignore
{{#include ../../tests/truss.rs:example}}
```

> Les conditions de Dirichlet **homogènes** (`u = 0`) ne nécessitent rien de
> plus que d'introduire la contrainte : la valeur imposée vaut 0 par défaut.
