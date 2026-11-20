# Macros Rust

Une macro sert ici à une seule chose : écrire **une fois** ce qui doit exister
en plusieurs exemplaires — la face Rust et la face Python d'un opérateur, la
même méthode sur les quatre types de champ, une ligne par type dans l'archive.
L'auteur écrit la physique, l'opérateur ou la liste ; la macro écrit la
répétition, et les exemplaires ne peuvent pas diverger.

Le dépôt en compte deux sortes :

- des `macro_rules!`, dans `src/` : elles reçoivent des morceaux de code et les
  recopient dans un gabarit ;
- une macro procédurale, `#[py_op]`, dans le paquet `macros/` : elle reçoit un
  item entier et le **lit** (voir plus bas pourquoi elle ne pouvait pas être
  une `macro_rules!`).

Cette page décrit celles qu'un contributeur invoque en ajoutant quelque chose.
Les autres ne servent qu'à l'intérieur de leur fichier (voir
[la fin de la page](#les-macros-internes-à-un-fichier)).

## Quelle macro pour quel ajout

| Pour ajouter… | Macro | Où l'invoquer |
|---|---|---|
| une physique | `physics_operator!` (forme « espace EF seul » ou « modèle cible ») | `src/models/<physique>.rs` |
| une loi de comportement | `physics_operator!` (forme « façade ») | `src/models/<physique>/<loi>.rs` |
| une énumération passée par son nom (`"TRI3"`, `"isotropic"`) | `named_enum!` | la liste de `src/named.rs` |
| la forme méthode d'un opérateur | `#[py_op]` | au-dessus du `#[pyfunction]`, dans `src/py/ops/<module>.rs` |
| un conteneur agrégat (côté Rust) | `impl_aggregate!`, `impl_aggregate_dump!` | `src/containers/<conteneur>.rs` |
| un conteneur agrégat (côté Python) | `impl_aggregate_pymethods!` | `src/py/<conteneur>.rs` |
| l'affichage d'un objet Python (`__str__`, `__repr__`, `dump`) | `impl_dump_pymethod!` | `src/py/<objet>.rs` |
| une fonction mathématique valeur par valeur sur un champ | `field_unary!`, `field_methods!`, `define_polymorphic_pyfunction!` | `src/ops/field/elementwise.rs`, `src/ops/field/methods.rs`, `src/py/ops/field.rs` |
| un type que l'archive sait écrire et relire | `shared!` ou `inline!`, `accessors!` ou `take_accessors!`, `inline_roots!` | `src/archive/registry.rs`, `src/archive/mod.rs` |

## `physics_operator!` — une physique, ses deux faces

Définie dans `src/models/mod.rs`. Invoquée dans le fichier de la physique, elle
émet l'opérateur public Rust — qui balaie l'espace EF, un sous-modèle par
sous-espace — et, sous la feature `python-api`, son `#[pyfunction]` avec son
attribut de stub, dans un module `<nom>_py`. L'auteur de la physique n'ouvre
jamais `src/py/`.

Elle a trois formes :

| forme | exemple de déclaration (dans le dépôt) | ce que la macro écrit |
|---|---|---|
| espace EF seul | `pub fn truss(fespace, strain_measure: StrainMeasure = StrainMeasure::Linear) via SubModel::truss;` | `truss(fespace, strain_measure)` en Rust ; en Python `strain_measure` devient un mot-clé avec sa valeur par défaut, et la doc Rust dit cette valeur |
| modèle cible | `pub fn radiation(fespace, target) via SubModel::radiation;` | la même chose, la cible passée au constructeur de chaque zone et ajoutée à la face Python |
| façade d'une loi | `pub fn drucker_prager(fespace, kinematics: Kinematics) = crate::ops::model::plasticity, PlasticLaw::DruckerPrager;` | une fonction qui délègue à l'opérateur générique (`plasticity` ou `damage`) avec la loi fixée, et l'argument `strain_measure` ajouté (`"linear"` par défaut en Python) |

Chaque forme prend deux documentations : les `///` Rust, avec leur doctest, et
un littéral `python: "…"` qui devient la docstring et atterrit dans le `.pyi`.
La forme façade complète ce littéral des sections communes à toutes les lois
(paramètres `fespace`, `kinematics`, `strain_measure`, retour, erreurs).

```rust
{{#include ../../../src/models/truss.rs:operator}}
```

Le détail — ce qu'il reste à raccorder à la main, et les formes qu'elle ne
couvre pas — est au chapitre [Ajouter une physique](../ajouter-une-physique.md).

## `#[py_op]` — la méthode d'un opérateur

Définie dans `macros/src/lib.rs`. Une fonction libre dont le premier argument
est le sujet est aussi exposée comme méthode de ce sujet (règle de
[Conventions](../conventions.md#le-verbe-exposé-aussi-en-méthode)). L'attribut
réécrit la fonction **inchangée**, puis ajoute au type `method_on` une méthode
qui l'appelle :

```text
#[py_op(method_on = PyMesh)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(signature = (mesh, angle_deg=None))]
pub fn skin(mesh: PyRef<PyMesh>, angle_deg: Option<f64>) -> PyResult<PyMesh>

    ↓ ajoute

#[pymethods]
impl PyMesh {
    // la documentation de la fonction, recopiée
    #[pyo3(signature = (angle_deg=None))]
    fn skin(slf: PyRef<'_, Self>, angle_deg: Option<f64>) -> PyResult<PyMesh>
}
```

- `method_on = PyType` est requis ; `name = "autre_nom"` donne à la méthode un
  autre nom que la fonction, quand le qualificatif du module doit passer dans
  le nom (`matrix::stiffness` → `stiffness_matrix`).
- Le receveur est déduit du sujet, et un `py: Python` en tête de la fonction
  est absorbé.
- La documentation et les `#[allow]` sont **recopiés**, pas pointés : la doc
  arrive ainsi dans `__doc__` **et** dans le `.pyi` que lisent les éditeurs, où
  un renvoi « See … » resterait du texte mort.
- L'attribut se pose **au-dessus** des autres : les macros d'attribut
  s'appliquent de l'extérieur vers l'intérieur, et celle-ci doit voir
  `#[pyfunction]` et `#[pyo3(signature = …)]` sur l'item qu'elle reçoit.

Pourquoi une macro procédurale : il faut **lire** une signature déjà écrite pour
en retirer le premier paramètre, et transformer
`#[pyo3(signature = (mesh, angle_deg=None))]` en la même liste sans sa
première entrée. Une `macro_rules!` ne voit que les morceaux qu'on lui passe et
ne peut pas ouvrir un fragment capturé ; une macro procédurale reçoit l'item
entier et le décompose.

## `named_enum!` — une valeur passée par son nom

Définie dans `src/named.rs`. Une énumération qui implémente le trait `Named`
(son vocabulaire : `VALUES`, `name()`, `aliases()`) traverse l'API Python comme
une **chaîne**. La macro écrit, pour chaque type de sa liste :

- la conversion **depuis** Python (`FromPyObject`) : une `str` lue par
  `Named::parse`, ou un `ValueError` qui dit ce qui n'a pas été compris et ce
  qui l'aurait été ;
- la conversion **vers** Python (`IntoPyObject`) : la `str` canonique, ce qui
  permet aussi à `#[pyo3(signature = …)]` d'écrire une valeur par défaut de ce
  type ;
- son type dans le stub (`PyStubType`) : `str`.

Un `#[pyfunction]` peut alors prendre le type Rust **directement** : sa
signature Python est sa signature Rust, et le wrapper n'a rien à décider.
Ajouter un type, c'est l'ajouter à la liste de l'invocation, à la fin de
`src/named.rs`.

## Les agrégats

Un agrégat (`Mesh`, `FiniteElementSpace`, `NodeField`, `ElementField`, `Model`,
`Matrix`, `Evolution`) est une liste de zones partagées. Ses accès, son union
`|` et son affichage sont les mêmes pour les sept, et viennent de quatre macros
de `src/aggregate.rs` :

| macro | invoquée sur | ce qu'elle écrit | précondition |
|---|---|---|---|
| `impl_aggregate!` | le type Rust (`Mesh`) | l'implémentation du trait `Aggregate` — l'accès à la liste des zones, dont le trait tire par défaut `len`, `get`, `iter`, `unit`, `union`, `union_sub`, `union_sub_first`, `add_sub`, `add_subs` —, puis `Index`, `IntoIterator`, `Debug`, `Display` et `union_subs` par `impl_aggregate_std_traits!` ; un bloc optionnel redéfinit des méthodes du trait (`check_push`, `post_push`, `display_extra`, `finalize`) | un membre `subs: Vec<Handle<Sub>>` |
| `impl_aggregate_dump!` | le type Rust | le trait `Dump` : le résumé `Display`, puis le `dump` de chaque zone sous un repère `── [i] ──` | la zone implémente `Dump` ; `Matrix` écrit le sien à la main |
| `impl_aggregate_pymethods!` | le wrapper Python (`PyMesh`) | `__len__`, `__getitem__`, `__or__` et `__ror__` sur l'agrégat, `__or__` sur la zone, `<zone>_count`, `<zone>(i)`, `add_<zone>`, les trois niveaux d'affichage, et les entrées du stub écrites à la main pour ces méthodes polymorphes | un membre `inner` qui implémente `Aggregate` |
| `impl_dump_pymethod!` | un wrapper Python qui n'est pas un agrégat | `__str__` (← `Display`), `__repr__` (← `Debug`), `dump(precision, max_rows, max_cols)` (← `Dump`) ; forme `handle` pour un membre `Handle<T>`, forme `value` pour une valeur possédée | le contenu implémente `Display`, `Debug` et `Dump` |

Les méthodes polymorphes d'agrégat (`__getitem__`, `__or__`, `__ror__`)
vivent dans des blocs `#[pymethods]` que le générateur de stub ne lit pas :
leurs entrées du `.pyi` sont écrites en Python dans les deux littéraux passés à
`impl_aggregate_pymethods!` (voir
[Compilation et tests](../compilation.md#les-dunders-polymorphes-des-agrégats)).
Ces blocs sont **fermés** : une méthode qu'on y ajouterait disparaîtrait du
stub sans un mot.

## Les champs

Les quatre types de champ — `NodeField`, `SubNodeField`, `ElementField`,
`SubElementField` — portent les mêmes opérations. Les macros les écrivent une
fois pour les quatre.

**Une fonction mathématique valeur par valeur** (`abs`, `sqrt`, `exp`, `log`,
`log10`, `cos`, `sin`, `tan`, `sinh`, `cosh`, `tanh`) en demande trois :

| macro | fichier | ce qu'elle écrit |
|---|---|---|
| `field_unary!` | `src/ops/field/elementwise.rs` | la fonction libre `ops::field::<nom>`, générique sur les quatre types de champ (trait `MapValues`), à partir de la fonction scalaire |
| `field_methods!` | `src/ops/field/methods.rs` | les méthodes Rust de pure délégation (`field.<nom>()`) ; invoquée une fois par type de champ, elle porte la liste des fonctions |
| `define_polymorphic_pyfunction!` | `src/py/ops/field.rs` | la fonction Python `pyrucast.field.<nom>`, qui reconnaît le type de champ reçu, et les quatre méthodes Python, toutes avec la même documentation |

**Les méthodes Python répétées** des wrappers de champ viennent de
`src/py/field_macros.rs`, une macro par forme et par famille (`impl_field_*`
pour `PyNodeField` / `PyElementField`, qui tiennent leur valeur dans `inner` ;
`impl_subfield_*` pour `PySubNodeField` / `PySubElementField`, qui la lisent par
`handle`) :

| macro | ce qu'elle écrit |
|---|---|
| `impl_pymethods_for_each!` | le même bloc de méthodes sur chaque type d'une liste, un `#[pymethods]` par type ; les autres macros de ce fichier s'appuient sur elle |
| `impl_field_transform_pymethod!` / `impl_subfield_transform_pymethod!` | une méthode qui rend un champ transformé du même type (`f.sqrt()`) |
| `impl_field_binary_pyslot!` / `impl_subfield_binary_pyslot!` | un opérateur binaire Python (`__add__`, `__rmul__`…), champ ∘ champ ou champ ∘ scalaire ; à invoquer dans le module qui déclare le `#[pyclass]` |
| `impl_field_mutator_pymethod!` / `impl_subfield_mutator_pymethod!` | une mutation d'une composante par un scalaire, en place (`add_to_component` et ses trois sœurs) |

**Les opérateurs Rust** (`+`, `-`, `*`, `/`) viennent de `src/containers/field.rs` :
`impl_subfield_scalar_ops!` (zone ∘ scalaire, le scalaire à gauche ou à
droite), `impl_subfield_field_ops!` (zone ∘ zone, par composante) et
`impl_field_ops!` (agrégat ∘ agrégat et agrégat ∘ scalaire). Chacune est
invoquée deux fois, pour la famille nodale et la famille aux points de Gauss.

## L'archive

`save` et `load` connaissent chaque type archivable par une **liste**, et
chaque liste est une invocation de macro : ajouter un type, c'est ajouter sa
ligne dans chacune. Le compilateur signale la variante de `Root` manquante.

| macro | fichier | une ligne par… | ce qu'elle écrit |
|---|---|---|---|
| `shared!` | `src/archive/registry.rs` | type partagé (`Coords` et les zones `Sub*`), qu'un `Handle` désigne | le décodage d'un enregistrement en `Handle<T>`, et sa remise comme objet nommé |
| `inline!` | `src/archive/registry.rs` | agrégat, tenu par valeur | le décodage de l'agrégat dans son entrée |
| `accessors!` | `src/archive/mod.rs` | type partagé | la méthode qui tire l'objet de la table relue (`objets.submesh("clef")`) |
| `take_accessors!` | `src/archive/mod.rs` | agrégat | la méthode qui **retire** l'agrégat de la table relue (`objets.mesh("clef")`) |
| `inline_roots!` | `src/archive/mod.rs` | agrégat | l'écriture de l'agrégat comme racine d'une sauvegarde |
| `value_accessors!` / `value_roots!` | `src/archive/mod.rs` | valeur simple (`bool`, `i64`, `f64`, `String` et leurs `Vec`) | la lecture et l'écriture de cette valeur |

Côté Python, `src/py/archive.rs` convertit chaque variante de `Root` en son
wrapper : un `match`, que le compilateur tient exhaustif.

## Les macros internes à un fichier

Le reste ne sert qu'à l'intérieur de son fichier, pour une répétition qui ne
sort pas de là :

- `forward_ref_binop!` (`src/containers/matrix.rs`) écrit, à partir de la forme
  `&a op &b` d'un opérateur de matrice, ses trois variantes qui prennent un
  opérande par valeur ;
- `__subfield_scalar_op!`, `__subfield_field_op!` et `__field_op!`
  (`src/containers/field.rs`) sont les gabarits par opérateur des trois macros
  d'opérateurs de champ ci-dessus.
