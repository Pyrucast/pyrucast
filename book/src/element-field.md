# Champ aux points de Gauss (`ElementField`)

Un **`ElementField`** porte une ou plusieurs valeurs **par `(cellule, point de Gauss)`** sur un [sous-espace éléments finis](fe-space.md). C'est le pendant de [`NodeField`](node-field.md) côté **valeurs aux points d'intégration**, et l'objet sur lequel s'écrivent naturellement :

- les **propriétés matériau** (module d'Young, coefficient de Poisson, conductivité, masse volumique, …) évaluées aux points où les intégrales sont calculées ;
- les **variables internes** (déformation plastique, endommagement, écrouissage, …) que l'on doit garder de cellule en cellule et de point en point ;
- les **grandeurs dérivées** d'une solution (contraintes, déformations, flux, …) pour le post-traitement.

## Support : un sous-espace éléments finis

Un `ElementField` est attaché à un seul `SubFiniteElementSpace` (cf. [Espace éléments finis](fe-space.md)) — celui-ci détermine :

- la liste des cellules concernées (via son `SubMesh`) ;
- le nombre de points de Gauss par cellule (via sa `QuadratureRule`).

Les trois dimensions de l'objet sont **figées à la construction** :

- `cell_count` — pris au `SubMesh::cell_count` du moment ;
- `gauss_count` — fourni par la quadrature du sous-espace ;
- `component_count` — choisi par l'utilisateur.

Le buffer interne est dimensionné une fois pour toutes et n'est **jamais réalloué**. La topologie du maillage sous-jacent doit rester figée pour la durée de vie du champ — c'est le contrat déjà documenté sur [`FiniteElementSpace`](fe-space.md) (les coordonnées peuvent évoluer, mais pas la connectivité ni le nombre de cellules).

Les coordonnées et poids des points de Gauss eux-mêmes ne sont **pas** stockés dans l'`ElementField` : ils restent sur le `SubFiniteElementSpace` comme données de référence, accessibles à la demande. L'`ElementField` ne contient que les valeurs utilisateur.

## Composantes nommées

Comme `NodeField`, un `ElementField` porte un ou plusieurs **noms de composantes** (`"E"`, `"nu"`, `"sigma_xx"`, `"plastic_strain"`, …). Les contraintes :

- au moins une composante à la construction ;
- noms uniques au sein d'un même champ ;
- valeurs initiales toutes à `0.0`.

## Disposition mémoire

Les valeurs sont rangées **à plat, ligne-major, dans l'ordre `cellule → gauss → composante`** :

```text
values[cell_idx * gauss_count * component_count
       + g * component_count
       + c]
```

Cet ordre rend deux accès courants particulièrement cache-friendly :

- lire **toutes les composantes** à un point de Gauss d'une cellule (par exemple `(E, nu, rho)` au point de Gauss courant pendant l'assemblage) — ce sont `component_count` flottants contigus ;
- balayer **tous les points de Gauss** d'une cellule pour une composante donnée — ce sont `gauss_count` flottants régulièrement espacés.

La méthode `point_values(cell, g)` expose directement le premier de ces patterns sous forme de slice.

## Refcount et cycle de vie

L'`ElementField` détient un `Handle<SubFiniteElementSpace>` (cloné, donc compté par référence). Tant que le champ est vivant, son sous-espace ne peut pas être collecté. Quand le `Drop` du champ s'exécute, le refcount du sous-espace est décrémenté ; s'il atteint zéro, la cascade descend jusqu'au `SubMesh` et à la `Configuration`.

`ElementField` lui-même n'incrémente **pas** le refcount des nœuds : il n'a pas de support nodal direct. Les nœuds restent protégés par le `SubMesh` du sous-espace, qui les incref déjà.

## API Rust

```rust,ignore
use pyrucast::mesh::configuration::Configuration;
use pyrucast::containers::element_field::ElementField;
use pyrucast::mesh::element_type::ElementType;
use pyrucast::finite_element_space::FiniteElementSpace;
use pyrucast::mesh::{Mesh, SubMesh};
use pyrucast::mesh::node::Node;
use pyrucast::store::{insert, with};

let cfg = insert(Configuration::new(2).unwrap());
let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
let c = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::TRI3));
mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
let sub = fes.subspace(0).unwrap();

// Élasticité linéaire 2-D : deux propriétés matériau.
let mut mat = ElementField::new(sub, vec!["E".into(), "nu".into()]).unwrap();
mat.set_uniform("E", 210e9).unwrap();    // module d'Young constant
mat.set_uniform("nu", 0.3).unwrap();     // coefficient de Poisson constant
assert_eq!(mat.value(0, 0, "E").unwrap(), 210e9);
```

Autres constructeurs et fillers utiles :

```rust,ignore
// Champ uniforme par composante en un appel.
let mat = ElementField::from_uniform_per_component(
    sub.clone(),
    vec!["E".into(), "nu".into(), "rho".into()],
    &[210e9, 0.3, 7800.0],
).unwrap();

// Valeur homogène sur une cellule (matériau par sous-domaine).
mat.set_cell_uniform(cell_idx, "E", 70e9).unwrap();

// Opérations scalaires sur une composante (par exemple, mise à l'échelle d'un module).
mat.mul_to_component("E", 0.95).unwrap();
```

Opérateurs avec un scalaire (le champ entier est translaté / mis à l'échelle) :

```rust,ignore
let scaled = &mat * 1.1;   // version par référence : préserve `mat`
let shifted = mat - 5.0;   // version consommante : zéro-copie
```

## API Python

```python
import pyrucast

# Maillage + FE space — préparation.
c = pyrucast.Configuration(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])
c2 = c.add_node([0.0, 1.0])
mesh = pyrucast.Mesh(c, "TRI3")
mesh.add_cell([a.id, b.id, c2.id])
fes = pyrucast.FiniteElementSpace(mesh)
sub = fes[0]

# Champ matériau.
mat = pyrucast.ElementField(sub, ["E", "nu"])
mat.set_uniform("E", 210e9)
mat.set_uniform("nu", 0.3)
print(mat)              # ElementField: 1 cell(s) × 3 gauss × 2 component(s) [E, nu]

# Accès par nom + indice.
assert mat.value(0, 0, "E") == 210e9
mat.set_value(0, 1, "nu", 0.28)

# Accès dictionnaire-like — `field[cell, gauss, "name"]`.
mat[0, 2, "E"] = 200e9
assert mat[0, 2, "E"] == 200e9

# Constructeur compact pour matériau homogène multi-composantes.
mat = pyrucast.ElementField.from_uniform_per_component(
    sub, ["E", "nu", "rho"], [210e9, 0.3, 7800.0],
)
```

## Sérialisation

`ElementField` implémente `Persist` via `serde` (comme tous les objets pyrucast). Le swap disque et la future sauvegarde fichier le traversent sans intervention de l'utilisateur ; seul le buffer de valeurs, la liste de noms et le `Handle<SubFiniteElementSpace>` voyagent dans le format binaire portable Linux ↔ Windows.

## Limitations actuelles

- L'`ElementField` est dimensionné par un seul `SubFiniteElementSpace`. Pour des grandeurs définies sur un maillage agrégé (`FiniteElementSpace`), il faudra construire un `ElementField` par sous-espace et les combiner côté utilisateur. Une variante agrégée pourra être ajoutée si le besoin se mesure.
- Pas encore d'opérations entre `ElementField` (addition champ + champ, contraction, etc.). À venir avec les premiers besoins concrets de l'assemblage et du post-traitement.
- Pas de mécanisme de **rééchantillonnage** entre quadratures (par exemple « projeter ce champ Gauss-2-points sur un autre Gauss-3-points »). Le sous-espace est implicitement figé à la création du champ.
