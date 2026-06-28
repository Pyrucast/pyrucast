# Matrice creuse (`Matrix`)

`Matrix` est le **conteneur de sortie** d'un assemblage : c'est ce que produisent les opérateurs `assemble::stiffness(model, materials)` / `assemble::mass(model)` à partir d'un [`Model`](model.md). Elle représente une matrice creuse dont les lignes et les colonnes sont identifiées par des **DOFs nommés**.

## Identification des DOFs : `(NodeId, nom_de_champ)`

Chaque ligne et chaque colonne d'une `Matrix` est identifiée par un couple `(NodeId, ChampId)` :

- **`NodeId`** — l'identifiant stable d'un nœud dans la `Coords`.
- **`ChampId`** — un indice compact dans une petite table de noms portée par la matrice (typiquement 5–10 entrées). Les noms sont des chaînes comme `"T"`, `"q"`, `"ux"`, `"lambda_w"`.

Le type concret est `DofId { node_id, field_idx }`. Cette représentation est compacte (un `u32` par champ partagé sur tous les DOFs qui le portent) et conserve l'information sémantique : à chaque entrée numérique de la matrice est attaché « quel inconnu, à quel nœud ».

Les jeux de DOFs de lignes et de colonnes sont **indépendants** :

- ils peuvent avoir des **tailles différentes** (matrice rectangulaire — par exemple le bloc Lagrange d'une condition de Dirichlet) ;
- ils peuvent porter des **noms de champs différents** (les lignes étiquetées par des duales `q`, les colonnes par des primales `T`).

## Stockage COO

Le format interne est **COO** (coordinate triplet list) : chaque insertion ajoute un triplet `(row_idx, col_idx, value)`. Plusieurs entrées au même `(row_idx, col_idx)` sont **conservées telles quelles** et **sommées à la lecture** ou à la mise en forme dense.

Conséquences :

- L'assemblage est **trivialement incrémental** : un `SubModel` peut appeler `add_entry(...)` autant de fois qu'il veut au même couple `(ligne, colonne)`, les contributions s'accumulent automatiquement.
- L'ordre des insertions est sans effet sur le résultat numérique final (commutativité de la somme).
- L'ordre des DOFs dans `row_dofs()` / `col_dofs()` est **l'ordre de première rencontre** lors des `add_entry`. C'est stable et reproductible pour une séquence d'assemblage donnée.

Le stockage propre à pyrucast reste COO pour la phase d'assemblage ; les opérations qui en profitent (matrice-vecteur, factorisation directe) utilisent `nalgebra-sparse` via des conversions à la demande :

- [`Matrix::to_csr`](#api-rust--accès-en-lecture) → `nalgebra_sparse::CsrMatrix<f64>`
- [`Matrix::to_csc`](#api-rust--accès-en-lecture) → `nalgebra_sparse::CscMatrix<f64>`
- [`Matrix::to_coo`](#api-rust--accès-en-lecture) → `nalgebra_sparse::CooMatrix<f64>`
- [`Matrix::to_dmatrix`](#api-rust--accès-en-lecture) → `nalgebra::DMatrix<f64>`

Cette stratégie « COO en construction, CSR/CSC à l'usage » colle au style cast3m (« finalisation au gel ») tout en évitant de réimplémenter ce que `nalgebra-sparse` fait déjà très bien.

## Drapeau `symmetric`

`Matrix::new(symmetric: bool)` accepte un drapeau qui déclare l'intention de l'assembleur :

- `true` : la matrice est numériquement symétrique (`A[i, j] = A[j, i]` pour les paires `(i, j)` correspondantes). C'est le cas de toute matrice de raideur d'une formulation variationnelle Galerkine standard.
- `false` : la symétrie n'est pas garantie (cas Lagrange seul, formulations non-Galerkine, problèmes de transport non self-adjoint, …).

**Le drapeau est informatif** : le stockage n'est **pas** dédupliqué (les deux triangles peuvent contenir des entrées indépendantes). Un solveur qui sait exploiter la symétrie (Cholesky) lit le drapeau pour décider de la factorisation ; un solveur générique l'ignore et utilise tout le contenu.

## Cas d'usage typique : matrice de raideur du laplacien

```rust,ignore
use pyrucast::containers::mesh::NodeId;
use pyrucast::containers::matrix::Matrix;

let mut k = Matrix::new(true);  // matrice de raideur, symétrique
// Modèle simple à 2 nœuds (segment) :
//   K = [[ 2, -1], [-1,  2]]
// Lignes étiquetées par la duale du modèle ("q" pour la thermique),
// colonnes par la primale ("T").
k.add_entry(NodeId(0), "q", NodeId(0), "T",  2.0);
k.add_entry(NodeId(0), "q", NodeId(1), "T", -1.0);
k.add_entry(NodeId(1), "q", NodeId(0), "T", -1.0);
k.add_entry(NodeId(1), "q", NodeId(1), "T",  2.0);

assert_eq!(k.n_rows(), 2);
assert_eq!(k.n_cols(), 2);
assert!(k.symmetric());
assert_eq!(k.dense(), vec![2.0, -1.0, -1.0, 2.0]);
```

## Matrice rectangulaire : bloc Lagrange

Une contrainte de Dirichlet introduit, par sa nature, un bloc **rectangulaire** : lignes indexées par les nœuds-multiplicateurs (un par contrainte), colonnes par les nœuds primaires contraints.

```rust,ignore
let mut c = Matrix::new(false);
// 2 contraintes : multiplicateur 100 lie le nœud 3, multiplicateur 101 lie le nœud 7.
c.add_entry(NodeId(100), "T", NodeId(3), "T", 1.0);
c.add_entry(NodeId(101), "T", NodeId(7), "T", 1.0);
assert_eq!(c.n_rows(), 2);
assert_eq!(c.n_cols(), 2);
// "T" est interné une seule fois dans la table de noms même s'il
// apparaît côté ligne ET côté colonne (la collision est résolue par
// les NodeIds distincts : 100 vs 3, 101 vs 7).
assert_eq!(c.field_names().len(), 1);
```

## API Rust — accès en lecture

```rust,ignore
// Valeur à une coordonnée (somme de toutes les entrées COO à ce point).
let v: f64 = k.get(NodeId(0), "q", NodeId(0), "T");

// Vue dense ligne-major (flat Vec, pratique pour Python).
let d: Vec<f64> = k.dense();
assert_eq!(d.len(), k.n_rows() * k.n_cols());

// Vue dense typée nalgebra (column-major DMatrix), prête pour LU/Cholesky.
let m: nalgebra::DMatrix<f64> = k.to_dmatrix();

// Vues creuses nalgebra-sparse, prêtes pour les solveurs creux.
let csr: nalgebra_sparse::CsrMatrix<f64> = k.to_csr();
let csc: nalgebra_sparse::CscMatrix<f64> = k.to_csc();

// Itération sur les triplets bruts (ordre d'insertion préservé).
for (row_dof, col_dof, value) in k.iter_entries() {
    let row_field = k.field_name(row_dof.field_idx);
    let col_field = k.field_name(col_dof.field_idx);
    // …
}

// Produit matrice-vecteur dense : y = A · x (passe par CSR sous le capot).
let y = k.mul_dense(&[1.0, 1.0]).unwrap();
```

## API Python

```python
import pyrucast

k = pyrucast.Matrix(symmetric=True)
k.add_entry(0, "q", 0, "T", 2.0)
k.add_entry(0, "q", 1, "T", -1.0)
k.add_entry(1, "q", 0, "T", -1.0)
k.add_entry(1, "q", 1, "T", 2.0)

assert k.n_rows() == 2
assert k.n_cols() == 2
assert k.symmetric is True

# Valeur ponctuelle.
assert k.get(0, "q", 0, "T") == 2.0

# Vue dense.
assert k.dense() == [2.0, -1.0, -1.0, 2.0]

# Tables de DOFs : (node_id, nom_du_champ).
assert k.row_dofs() == [(0, "q"), (1, "q")]
assert k.col_dofs() == [(0, "T"), (1, "T")]

# Matrice-vecteur.
y = k.mul_dense([1.0, 1.0])
assert y == [1.0, 1.0]

# Itération brute sur les triplets (ordre d'insertion).
for row_node, row_field, col_node, col_field, value in k.entries():
    pass
```

## Sérialisation

`Matrix` implémente `Persist` via `serde` (comme tous les objets pyrucast). Les triplets COO, la table de noms et les DOFs voyagent dans le format binaire portable Linux ↔ Windows.

## Limitations actuelles

- **COO interne uniquement pendant l'assemblage** : `add_entry` n'utilise pas (encore) le `CooMatrix` de `nalgebra-sparse` parce que celui-ci exige des dimensions fixes à la construction ; or pyrucast découvre les DOFs au fil de l'insertion. La conversion vers `CooMatrix` / `CsrMatrix` / `CscMatrix` est faite à la demande au moment où ces vues sont utiles.
- **Recherche linéaire des DOFs et des noms** lors de l'insertion : O(n_dofs + n_fields) par `add_entry`. Pour les premiers besoins (assemblage de quelques milliers de DOFs), c'est négligeable. Une indexation hash pourra être ajoutée en Phase 6 sans casser l'API publique.
- **Pas de produit matrice-matrice** ni d'opérations entre matrices (somme, etc.) : à venir avec les premiers besoins concrets (préconditionneurs, formulations couplées).
- Le drapeau `symmetric` n'est pas vérifié numériquement à l'assemblage. C'est de la responsabilité de l'assembleur (du `Model`) d'apparier correctement la déclaration et la réalité.
