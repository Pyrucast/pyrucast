# Champ aux nœuds (`NodeField` / `SubNodeField`)

Un champ aux nœuds porte une ou plusieurs valeurs **par nœud**. Il suit la
même grammaire d'agrégat que tous les conteneurs de pyrucast (cf.
[Conventions](conventions.md)) :

- **`SubNodeField`** — les valeurs d'**une zone** : un bloc multi-composantes
  sur les nœuds d'un sous-maillage POI1 (cf. [Maillage](mesh.md)) ;
- **`NodeField`** — l'**agrégat** : une liste de `SubNodeField`, un par zone,
  avec éventuellement des **composantes différentes d'une zone à l'autre**.

C'est le miroir exact de `ElementField` / `SubElementField` côté valeurs aux
nœuds. L'intérêt de l'agrégat : un champ multiphysique (par exemple `T` sur
tout le domaine, `UX`/`UY` sur la zone solide seulement) se représente sans
inventer de `0.0` pour les couples `(nœud, composante)` qu'aucune zone ne
définit — rien n'est densifié.

## Support : un sous-maillage POI1 par zone

Un `SubMesh` POI1 est, par construction, **exactement une liste de nœuds**
(un nœud par cellule). Chaque `SubNodeField` s'appuie sur un support POI1 :

- construit depuis un SubMesh POI1, le champ **partage le handle** du
  support tel quel (aucun refcount par nœud supplémentaire : le SubMesh est
  l'unique propriétaire des increfs, garder son handle suffit à garder les
  nœuds en vie) ;
- construit depuis un SubMesh d'un autre type d'élément, le champ
  matérialise un support POI1 frais sur les **nœuds distincts** de la zone
  (ordre de première apparition dans la connectivité).

La liste de nœuds est **figée à la construction** (la connectivité d'un
SubMesh est gelée par convention de projet) : la taille du buffer de
valeurs ne change jamais.

```text
   NodeField (agrégat)
   ├── SubNodeField zone 0 ── support POI1 ── values[i × ncomp + c]
   ├── SubNodeField zone 1 ── support POI1 ── values[...]
   └── …
```

## Composantes nommées

Chaque zone porte ses **noms de composantes** (`"UX"`, `"UY"`, `"T"`, …),
rangés en **row-major** : la composante `c` du nœud `i` est à l'indice
`i × ncomp + c`. Au moins une composante par zone, noms uniques, valeurs
initialisées à `0.0`. Au niveau agrégat, `components()` renvoie l'**union**
des composantes des zones (ordre de première apparition).

Les caractéristiques communes à tous les champs (composantes, `min`, `max`,
`sum`, arithmétique scalaire et par composante) sont portées par les traits Rust
[`SubField` (niveau zone) et `Field` (niveau agrégat)](field.md) — partagés
avec [`ElementField`](element-field.md).

## Nœuds d'interface : duplication, lecture, cohérence

Un nœud partagé par plusieurs zones (nœud d'interface) est stocké **une
fois par zone**. Trois règles régissent cette duplication :

- **lecture agrégat** (`field.value(nœud, comp)`) : la **première zone**
  définissant le couple gagne — aucune vérification au fil de l'eau. La
  forme par lot `field.values(nœuds, comp)` lit une **liste** de valeurs
  dans le **même ordre** : `nœuds` est une liste de nœuds, un `SubMesh`
  POI1, ou un `Mesh` POI1 (ses points pris dans l'ordre de la
  connectivité) ; même règle « première zone » et même erreur qu'un nœud
  non défini ;
- **écriture** : il n'y a **pas** d'écriture au niveau agrégat ; toute
  mutation passe par les zones (`field[i]`), exactement comme
  `ElementField` ;
- **cohérence à la demande** : `field.check()` vérifie que toutes les
  zones stockant un même couple `(nœud, composante)` portent la **même**
  valeur (comparaison exacte) ; `node_field.consolidate(field)` fait cette
  vérification puis **fusionne par support** — les zones définies sur le
  **même** `SubMesh` (identité de handle) deviennent une seule zone portant
  l'union de leurs composantes (valeurs des composantes communes vérifiées),
  les supports distincts restent séparés.

## Composition : union `|`

Comme pour tous les agrégats, `a | b` **unit les zones** (handles partagés,
pas de copie ; déduplication par handle) — ce n'est pas une addition de
valeurs. L'union **finalise** en fusionnant les zones de même support (voir
`node_field.consolidate` ci-dessus) et lève si deux zones divergent sur une valeur
partagée. L'arithmétique scalaire (`f + 2.0`, `f * 0.5`, …) vit au niveau
zone (`SubNodeField`) sur `+`/`*`/… Le nommé `merge(a, b)` ≡ `a | b`.

## API Rust

```rust,ignore
use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node};
use pyrucast::coords::Coords;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::node_field::NodeField;
use pyrucast::handle::Handle;

let coords = Handle::new(Coords::new(2).unwrap());
let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
let b = Node::create_in(coords.clone(), &[1.0, 0.0]).unwrap();

// Support : SubMesh POI1 contenant a et b.
let sm = {
    let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
    sm.add_cell(&[a.id()]).unwrap();
    sm.add_cell(&[b.id()]).unwrap();
    Handle::new(sm)
};

// Champ de déplacement 2D mono-zone : composantes UX, UY.
let u = NodeField::from_submesh(&sm, vec!["UX".into(), "UY".into()]).unwrap();

// Écriture : via la zone. Lecture : via l'agrégat (ou la zone).
write(&u.get(0).unwrap()).unwrap().set_value(a.id(), "UX", 1.5).unwrap();
assert_eq!(u.value(a.id(), "UX").unwrap(), 1.5);
assert_eq!(u.value(b.id(), "UX").unwrap(), 0.0);   // valeur par défaut

// Depuis un maillage multi-zones : un SubNodeField par submesh.
let field = NodeField::new(&mesh, vec!["T".into()]).unwrap();
assert_eq!(field.len(), mesh.len());
field.check().unwrap();   // zones cohérentes aux interfaces
```

## API Python

```python
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])

mesh = pyrucast.Mesh(c, "POI1")
mesh.unit().add_cell([a])
mesh.unit().add_cell([b])

# Un SubNodeField par submesh du support (Mesh ou SubMesh).
u = pyrucast.NodeField(mesh, ["UX", "UY"])
print(u)  # NodeField: 1 subfield(s)
print(u.unit())  # SubNodeField: 2 node(s), 2 component(s) [UX, UY]

# Écriture via la zone, lecture via l'agrégat.
u[0][a, "UX"] = 1.5
print(u.value(a, "UX"))  # 1.5

# Lecture par lot : liste de nœuds (ou Mesh/SubMesh POI1) → liste ordonnée.
print(u.values([a, b], "UX"))  # [1.5, 0.0]
print(u.values(mesh, "UX"))  # [1.5, 0.0]  — points du maillage POI1
print(u.min("UX"), u.max("UX"))  # 0.0 1.5
print(u.sum("UX"))  # 1.5  — Σ sur les nœuds (résultante d'un champ de forces)

# Composantes par zone (multiphysique) :
f = pyrucast.NodeField.with_components_per_submesh(two_zone_mesh, [["T"], ["UX", "UY"]])
print(f.components())  # ['T', 'UX', 'UY']
f.check()  # cohérence des interfaces (lève sinon)
g = pyrucast.node_field.consolidate(f)  # fusion au plus juste
```

## Refcount et durée de vie

Le champ ne fait **aucune** comptabilité par nœud : il garde un clone du
`Handle<SubMesh>` de son support, et c'est le SubMesh qui possède les
increfs par nœud dans la `Coords` (cf.
[Coords](coords.md)). Tant qu'une zone du champ est vivante,
son support l'est aussi, donc ses nœuds aussi — même si tous les `Node`
utilisateurs ont disparu.

La libération est automatique : quand le dernier `Handle` sur une zone
disparaît, la zone est détruite, son clone du handle de support avec elle, et
les nœuds redeviennent collectables si plus rien ne les retient (cf.
[Modèle mémoire](memory-model.md)).

## Opérateurs consommant un champ

Les opérateurs détaillés sont décrits dans
[Opérateurs sur les champs](operateurs/champs.md) (dérivations),
[Assemblage](operateurs/assemblage.md) (`flux`, second membre) et
[Solveur](operateurs/solveur.md). Tous consomment l'agrégat et résolvent les
nœuds **à travers les zones** (règle premier-trouvé) :

| Opération | Particularité multi-zones |
|---|---|
| `positions(mesh)` | un `SubNodeField` par submesh, interfaces cohérentes par construction |
| `coords.set(f)` / `displace(f)` | chaque nœud distinct traité **une seule fois** (un nœud d'interface n'est pas déplacé deux fois) |
| `gradient(f, fes)` / `deformation(u, fes)` | lookups par nœud × Gauss via un snapshot des zones |
| `divergence(F)` | adjoint de `gradient` : champ vectoriel par éléments → `NodeField` (`div`), accumulé par nœud (`d_i = ∫ ∇N_i·F`) |
| `solve(matrix, rhs)` | second membre lu par DOF (absent ⇒ `0.0`) ; solution : une zone par support **colonne** des blocs de la matrice, sur le handle même du bloc — `same_support` avec tout champ posé sur ces supports, stable d'une résolution à l'autre |
| `restrict(f, mesh)` | une zone par submesh cible, sur le **nuage POI1 canonique caché** du sous-maillage (`to_poi1`) ⇒ deux restrictions sur le même `mesh` partagent le support et sont soustractibles (et s'alignent avec `K·x`/`solve`) ; `0.0` pour les nœuds non couverts |
| `restrict_like(f, target)` | reprojette sur le support **et** les composantes de `target` (mêmes slots) ⇒ combinable par `+ - * /` avec `target` ; nœuds/composantes hors de `target` abandonnés, `0.0` si non couverts |
| `merge(a, b)` | union structurelle consolidée (conflit de valeur ⇒ erreur) |
| `node_field.consolidate(f)` | fusion par jeu de composantes après vérification de cohérence |
