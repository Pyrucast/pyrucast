# Champ aux points de Gauss (`ElementField` / `SubElementField`)

Un champ aux points de Gauss porte une ou plusieurs valeurs **par `(cellule,
point de Gauss)`** sur un [espace éléments finis](fe-space.md). Il suit la même
grammaire d'agrégat que tous les conteneurs de pyrucast (cf.
[Agrégat](aggregate.md)) et le contrat de [champ](field.md) :

- **`SubElementField`** — les valeurs d'**une zone** : un bloc multi-composantes
  sur les cellules × points de Gauss d'**un** `SubFiniteElementSpace` ;
- **`ElementField`** — l'**agrégat** : une liste de `SubElementField`, un par
  sous-espace, avec éventuellement des **composantes différentes d'une zone à
  l'autre**.

C'est le miroir exact de [`NodeField`](node-field.md) côté **points
d'intégration**. C'est l'objet sur lequel s'écrivent naturellement :

- les **propriétés matériau** (module d'Young, Poisson, conductivité, masse
  volumique…) évaluées là où les intégrales sont calculées ;
- les **variables internes** (déformation plastique, endommagement…) gardées de
  cellule en cellule et de point en point ;
- les **grandeurs dérivées** d'une solution (contraintes, déformations, flux…)
  pour le post-traitement, produites par les [opérateurs de
  champ](operateurs/champs.md) (`gradient`, `deformation`…) et le
  [comportement](operateurs/comportement.md) (`integrate_behavior`).

## Support : un sous-espace éléments finis par zone

Chaque `SubElementField` est attaché à un seul `SubFiniteElementSpace` (cf.
[Espace éléments finis](fe-space.md)), qui détermine :

- la liste des cellules concernées (via son `SubMesh`) ;
- le nombre de points de Gauss par cellule (via sa `QuadratureRule`).

Les trois dimensions d'une zone sont **figées à la construction** :
`cell_count` (du `SubMesh`), `gauss_count` (de la quadrature),
`component_count` (choisi par l'utilisateur). Le buffer interne est dimensionné
une fois pour toutes et n'est **jamais réalloué** — la topologie du maillage
sous-jacent doit rester figée pour la durée de vie du champ (les coordonnées,
elles, peuvent évoluer ; cf. [`FiniteElementSpace`](fe-space.md)).

Les coordonnées et poids des points de Gauss ne sont **pas** stockés dans le
champ : ils restent sur le `SubFiniteElementSpace` comme données de référence.

```text
   ElementField (agrégat)
   ├── SubElementField zone 0 ── support SubFiniteElementSpace ── values[…]
   ├── SubElementField zone 1 ── support SubFiniteElementSpace ── values[…]
   └── …
```

## Composantes nommées

Chaque zone porte ses **noms de composantes** (`"E"`, `"nu"`, `"sigma_xx"`,
`"plastic_strain"`…) : au moins une, noms uniques, valeurs initialisées à
`0.0`. Au niveau agrégat, `components()` renvoie l'**union** des composantes
des zones (ordre de première apparition) ; une composante peut n'exister que
sur certaines zones.

## Disposition mémoire

Les valeurs d'une zone sont rangées **à plat, ligne-major, dans l'ordre
`cellule → gauss → composante`** :

```text
values[cell_idx * gauss_count * component_count
       + g * component_count
       + c]
```

Cet ordre rend deux accès courants cache-friendly :

- lire **toutes les composantes** à un point de Gauss d'une cellule (par
  exemple `(E, nu, rho)` pendant l'assemblage) — `component_count` flottants
  contigus, exposés par `point_values(cell, g)` ;
- balayer **tous les points de Gauss** d'une cellule pour une composante donnée
  — `gauss_count` flottants régulièrement espacés.

## Construction : au niveau agrégat

Comme tout agrégat, un `ElementField` se construit au **niveau parent**, à
partir d'un `FiniteElementSpace` : une zone par sous-espace.

- `ElementField(fes, components)` — la **même** liste de composantes pour
  chaque sous-espace ;
- `ElementField.with_components_per_subspace(fes, [...])` — une liste de
  composantes **par** sous-espace (multiphysique / multi-matériau).

Pour fabriquer un champ matériau prêt pour l'assemblage, l'opérateur
[`material_field(model, [...])`](operateurs/construction.md) est plus direct : il
crée les zones nécessaires aux sous-modèles qui consomment du matériau et les
remplit en un appel.

## Refcount et cycle de vie

Chaque zone détient un `Handle<SubFiniteElementSpace>` (cloné, donc compté).
Tant qu'une zone est vivante, son sous-espace ne peut pas être collecté ; à son
`Drop`, le refcount du sous-espace décroît et la cascade descend jusqu'au
`SubMesh` puis à la `Coords`. Un `ElementField` n'incrémente **pas** le
refcount des nœuds : il n'a pas de support nodal direct — les nœuds restent
protégés par le `SubMesh` du sous-espace.

## API Rust

```rust,ignore
use pyrucast::aggregate::Aggregate;
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::field::{Field, SubField};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::coords::Coords;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::handle::Handle;

let coords = Handle::new(Coords::new(2).unwrap());
let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();
let c = Node::create_in(coords.clone(), &[0.0, 1.0]).unwrap();
let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

// Élasticité linéaire 2-D : deux propriétés matériau, une zone (un sous-espace).
let mat = ElementField::new(&fes, vec!["E".into(), "nu".into()]).unwrap();
{
    let mut z = write(&mat.get(0).unwrap()).unwrap();   // la zone (SubElementField)
    z.set_uniform("E", 210e9).unwrap();                 // module d'Young constant
    z.set_uniform("nu", 0.3).unwrap();                  // Poisson constant
    assert_eq!(z.value(0, 0, "E").unwrap(), 210e9);
}

// Composantes par sous-espace (multi-matériau) :
let mat2 = ElementField::with(
    &fes,
    &[vec!["E".into(), "nu".into()]],   // une liste par sous-espace
).unwrap();

// Statistiques et arithmétique au niveau agrégat.
assert_eq!(Field::max(&mat, "E").unwrap(), 210e9);
let scaled = &mat * 1.1;       // nouveau champ (référence : préserve `mat`)
mat.mul_to_component("E", 0.95).unwrap();   // en place, seulement "E"
```

## API Python

```python
{{#include ../../tests/python/test_doc_conteneurs.py:element_field_api}}
```

Le plus souvent, on ne construit pas le champ matériau à la main : on appelle
l'opérateur [`material_field`](operateurs/construction.md) qui apparie les zones aux
sous-modèles consommateurs de matériau et ignore les autres (Dirichlet, …).

## Visualisation

`element_field.plot(...)` colore le champ **sur son propre support** : chaque
zone retrouve son sous-maillage via son sous-espace EF (partagé, pas copié). Le
rendu raisonne **par élément** — les valeurs nodales viennent d'un moindre
carré local à l'élément sur les valeurs de Gauss, sans moyenne inter-éléments,
de sorte que les **discontinuités (flux, contraintes) restent visibles**. Voir
[Visualisation](visualization.md).

## Sérialisation

`SubElementField` (et donc `ElementField`) implémente `Portable` via `serde`
comme tous les objets pyrucast : le buffer de valeurs et la liste de noms
voyagent dans le format binaire portable Linux ↔ Windows. Le lien vers l'espace
EF, lui, est un `Handle` : il devient un identifiant local au fichier, et
l'espace est écrit avec le champ — voir [Sauvegarde et
relecture](sauvegarde.md).

## Limitations actuelles

- Pas de mécanisme de **rééchantillonnage** entre quadratures (« projeter ce
  champ Gauss-2-points sur un autre Gauss-3-points ») : le sous-espace est figé
  à la création de chaque zone.
- L'arithmétique binaire entre champs est **stricte** (même support, mêmes
  composantes ; cf. [Champ](field.md)) — il n'y a pas (encore) de combinaison
  tolérante avec rééchantillonnage ou complétion par zéro.
