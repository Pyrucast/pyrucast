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

## Blocs bi-mode : littéral ou calculé

Une `Matrix` est un **agrégat de blocs** `SubMatrix`, et un bloc est de l'un de deux modes :

- **littéral** — il porte ses **valeurs**, stockées en **COO** (coordinate triplet list). Chaque `add_entry(...)` ajoute un triplet `(ligne, colonne, valeur)` ; plusieurs entrées au même couple s'**accumulent** (sommées à l'assemblage), l'ordre d'insertion étant sans effet. C'est le mode historique — celui des contraintes (blocs `C` / `Cᵀ` de Dirichlet) et de tout bloc monté à la main.
- **calculé** — il ne porte **aucune valeur**, seulement une **recette** `{ sous-modèle, sous-espace EF, matériau }`. Ses entrées sont produites **à l'assemblage** par le noyau élémentaire du sous-modèle, dispersées directement dans la matrice globale. C'est le mode des physiques volumiques (raideur), qui évite de matérialiser un COO intermédiaire.

Un bloc calculé garde son lien vers sa physique **via la recette** ; la `Matrix`, elle, reste un simple sac de blocs et **ne référence pas le `Model`**.

### Étiquette de nature physique (`physics`)

Chaque bloc porte en plus un **ensemble de natures** `Vec<Physics>` (`Mechanical`,
`Thermal`, `Constraint`, `Other`) — l'assembleur le pose sur **tout** bloc qu'il
émet, sur les deux chemins (calculé **et** littéral), donc le couple `C`/`Cᵀ` d'un
Dirichlet est étiqueté lui aussi. C'est ce qui rend l'étiquette utilisable là où la
recette manque (blocs littéraux). Le tag est un **ensemble** : vide pour un bloc
monté à la main hors assemblage (le cas « rien »), et à plusieurs éléments pour
une physique couplée.

Il alimente `Matrix::filter(Physics)` — le miroir de
[`Model::filter`](model.md#nature-physique-et-filtrage) — qui renvoie une `Matrix`
ne gardant que les blocs dont l'ensemble **contient** la nature donnée (handles
partagés, pas de copie). Le résultat n'est **pas** assemblé : relancer
`ops::assemble::assemble` avant de résoudre. Un bloc à l'ensemble vide n'est
sélectionné par aucune nature concrète ; l'étiqueter `Physics::Other` le rend
atteignable. `Matrix::physics()` renvoie l'ensemble des natures présentes dans la
matrice (dédupliqué — « plusieurs tags » au niveau de l'agrégat).

```rust,ignore
let k_meca = k.filter(Physics::Mechanical)?;   // blocs au moins mécaniques
let natures = k.physics()?;                     // ex. [Thermal, Constraint]
```

## Assemblage : motif + scatter

Passer d'un agrégat de blocs à une matrice utilisable se fait en deux temps :

1. **Motif creux** (sparsité CSR) — l'**union dédoublée** des DDL des blocs (une liste globale, via table de hachage) et de leurs entrées. Il ne dépend que de la **topologie** (bloc calculé via la connectivité, bloc littéral via sa COO), pas des matériaux ; `stiffness` le **mémoïse donc sur le `Model`** et le réutilise d'un assemblage à l'autre.
2. **Valeurs** — dispersées (scatter) dans le CSR : un bloc calculé lance son noyau élémentaire (en parallèle, par coloration des cellules — voir [Parallélisme](developper/parallelisme.md)) ; un bloc littéral recopie sa COO. Chaque bloc remappe sa numérotation locale `(nœud, variable)` vers l'index global via une **table de traduction** — O(nnz), sans recherche par entrée (`NodeId` est déjà l'index nœud global dense, et `add_entry` retrouve la position d'un nœud en O(1)).

L'ordre des DOFs dans `row_dofs()` / `col_dofs()` est l'**ordre de première rencontre** des blocs — sauf si le `Coords` porte une [`permutation`](coords.md) (ordre solveur), auquel cas la liste globale suit cet ordre (tri stable). Reproductible dans les deux cas.

### `finalize` vs `ops::assemble`

- `Matrix::finalize()` n'assemble que des blocs **littéraux** (somme des COO → CSR). Il **refuse** un bloc calculé : le noyau vit dans `models`, hors de `containers`, et l'y appeler créerait un cycle `matrix ↔ kernel`. Il renvoie alors vers `ops::assemble`.
- `ops::assemble::stiffness(model, materials)` construit les blocs (calculés pour les physiques volumiques, littéraux pour Dirichlet) et assemble, motif mémoïsé sur le `Model`.
- `ops::assemble::assemble(&mut m)` réassemble une matrice **depuis ses blocs seuls**, sans `Model` : c'est le chemin de **composition** — combiner une sous-matrice neuve (de provenance quelconque) à une matrice existante puis réassembler. La `Matrix` ne dépendant que de ses blocs, cette composabilité de base est ainsi préservée y compris en présence de blocs calculés.

Les opérations qui profitent du creux (matrice-vecteur, factorisation directe) utilisent `nalgebra-sparse` via des conversions à la demande :

- [`Matrix::to_csr`](#api-rust--accès-en-lecture) → `nalgebra_sparse::CsrMatrix<f64>`
- [`Matrix::to_csc`](#api-rust--accès-en-lecture) → `nalgebra_sparse::CscMatrix<f64>`
- [`Matrix::to_coo`](#api-rust--accès-en-lecture) → `nalgebra_sparse::CooMatrix<f64>`
- [`Matrix::to_dmatrix`](#api-rust--accès-en-lecture) → `nalgebra::DMatrix<f64>`

## Facteur scalaire (`Mul<f64>` / `Div<f64>`) et combinaison de matrices

Chaque `SubMatrix` porte un **facteur** `f64`, `1.0` par défaut, multiplié ou divisé par
`bloc * s` / `bloc / s`. Le facteur ne touche **que** ce champ — jamais les valeurs
stockées (`coo`) — ce qui le rend utilisable aussi bien sur un bloc **littéral** que sur
un bloc **calculé** (dont les valeurs n'existent qu'à l'assemblage, produites par le
noyau élémentaire). Il est pris en compte partout où une valeur du bloc est lue ou
émise : les accesseurs directs (`get`, `dense`, `to_dmatrix`, `to_coo`, `to_csr`,
`to_csc`, `mul_dense`) et les deux passes d'assemblage global (`Matrix::finalize` et
`ops::assemble::scatter`, calculé comme littéral). Seules les formes **locales** brutes
(`local_triplets`, `local_coo_arrays`) restent non mises à l'échelle — ce sont des vues
internes destinées au remappage global, chaque consommateur y applique le facteur
lui-même.

`&Matrix * s` / `&Matrix / s` mettent à l'échelle une matrice entière : chaque bloc est
**cloné** dans un nouveau slot de store avec son facteur ajusté — jamais muté en place.
C'est nécessaire car `add_sub`/`union`/`filter`/`subset` **partagent** les
`Handle<SubMatrix>` (refcount, même slot) plutôt que de les copier ; muter le facteur
en place risquerait de rescaler silencieusement toute autre `Matrix` référençant le
même bloc. Comme pour `filter`, le résultat n'est **pas assemblé** — `finalize()` ou
`ops::assemble::assemble()` avant de résoudre.

```rust,ignore
let m_dt = (&m / dt)?;      // facteur = 1/dt sur chaque bloc de M, aucune valeur réécrite
assert_eq!(m.get(a, "q", a, "T"), m_dt.get(a, "q", a, "T") * dt); // m inchangée
```

**Pas d'opérateur `Matrix + Matrix`** : l'assembleur somme déjà les contributions qui
tombent sur le même `(row, col)` global (`build_global_triplets`,
`scatter_serial`/`scatter_parallel`). `M/dt + K` s'obtient donc avec les primitives
existantes — l'union `|` (partage de blocs, pas de copie) suivie d'un réassemblage :

```rust,ignore
let mut sys = (&(&m / dt)? | &k)?;
ops::assemble::assemble(&mut sys)?;   // requis dès qu'un bloc calculé est présent
let rhs = (&f_ext + &(&m_dt * &u0)?)?;
let u = solver::lu::solve(&sys, &rhs)?;
```

Aucun traitement particulier n'est nécessaire quand `K` et `M` n'ont pas le même
ensemble de DOFs (cas courant : un Dirichlet/MPC n'entre que dans la matrice de
raideur, jamais dans la masse) — l'union prend simplement l'union des DOFs des deux
côtés, et les blocs de `M` ne contribuent rien aux DOFs qu'ils ne portent pas.

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

// Produit matrice · champ : `x` est lu aux DOFs *colonnes* (vars **primales** ;
// une entrée qu'aucune zone ne définit vaut 0), le résultat est un `NodeField`
// sur les DOFs *lignes* (vars **duales**) — `K · u = f` — avec une zone par
// support **ligne** des blocs, sur le handle du bloc lui-même (`same_support`
// garanti avec tout champ posé sur ces supports). L'opérateur `*` (`&k * &x`)
// est le sucre syntaxique de `mul_field`. C'est le miroir exact de
// `solver::solve`, qui lit un champ dual aux lignes et rend un champ primal aux colonnes.
let y: NodeField = k.mul_field(&x).unwrap();
let y: NodeField = (&k * &x).unwrap();

// Maillages des supports des blocs (handles partagés, dédupliqués ; disponibles
// avant `finalize`) : `row_mesh` côté dual (lignes), `col_mesh` côté primal
// (colonnes). Cible de projection pour poser un champ sur les supports mêmes
// des sorties de `mul_field` / `solve` et le combiner zone à zone :
let f_ext_r = restrict(&f_ext, &k.row_mesh().unwrap()).unwrap();
let r = (&f_ext_r - &y).unwrap();     // s'aligne par support, pas de passthrough de zone
// ⚠ une composante absente d'un côté passe brute (union) ; pour un résidu
// strict (toute composante soustraite, absente lue à 0), reprojeter sur le
// support ET les composantes de l'autre opérande :
let r = (&restrict_like(&f_ext, &y).unwrap() - &y).unwrap();
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

# Matrice · champ : `k * x` lit `x` aux DOFs colonnes et rend un NodeField
# neuf sur les nœuds des DOFs lignes.
y_field = k * x  # x: pyrucast.NodeField

# Itération brute sur les triplets (ordre d'insertion).
for row_node, row_field, col_node, col_field, value in k.entries():
    pass

# Facteur scalaire — chaque bloc a un `.factor` en lecture seule (1.0 par défaut).
assert k[0].factor == 1.0

# `matrix * scalaire` / `matrix / scalaire` : une Matrix neuve, blocs clonés avec
# leur facteur ajusté (aucune valeur réécrite) ; k reste inchangée.
m_dt = m / dt
assert m[0].factor == 1.0  # m inchangée
assert m_dt[0].factor == 1.0 / dt

# `matrix * NodeField` (produit matrice-vecteur) coexiste avec `matrix * scalaire` :
# le type de l'opérande de droite détermine laquelle des deux opérations s'exécute.
sys = m_dt | k
sys.finalize()  # (ou `pyrucast.assemble.assemble(sys)` si des blocs sont calculés)
```

## Sérialisation

`Matrix` implémente `Persist` via `serde` (comme tous les objets pyrucast). Les triplets COO, la table de noms et les DOFs voyagent dans le format binaire portable Linux ↔ Windows.

## Limitations actuelles

- **Cache de motif non invalidé par les mutations profondes** : le motif creux mémoïsé sur le `Model` est invalidé à l'ajout d'un sous-modèle (`add_sub`), mais pas si le maillage / l'espace EF sous-jacent change *en place* (remaillage) — reconstruire le modèle dans ce cas. Le chemin de composition `ops::assemble::assemble(&mut m)`, lui, reconstruit toujours le motif depuis les blocs.
- **Pas de produit matrice-matrice** ni d'opérations algébriques entre matrices (somme, etc.) : à venir avec les premiers besoins concrets (préconditionneurs, formulations couplées).
- Le drapeau `symmetric` n'est pas vérifié numériquement à l'assemblage. C'est de la responsabilité de l'assembleur (du `Model`) d'apparier correctement la déclaration et la réalité.
