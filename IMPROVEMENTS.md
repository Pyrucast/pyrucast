# Improvements for later

One heading per item, followed by what is needed to pick it up. An item leaves
this file once it is done.

## Two-field operations copy the right-hand field first

`Field::merge_field`, `dot_field` and `pscal_field` (`src/containers/field.rs`)
start by cloning every zone of `other`; the Python `binary` dispatcher of
`PyNodeField` (`src/py/node_field.rs`) clones a `SubNodeField` operand the same
way.

- Why: `other` may share zone handles with `self` (`u + u`), whose zones are
  read in place under read guards. A second read guard on the same handle can
  deadlock: parking_lot's `RwLock` is task-fair, a waiting writer blocks it.
- Cost (2026-11-02, two POI1 zones splitting a line of 60 M nodes, 3
  components, 4 threads): `u + v` takes 0.68 s, about 0.31 s of it in this
  single-threaded copy. The addition itself, about 0.37 s, matches numpy's
  `a + b` on one thread: it is memory-bound. The copy is about 15 % of `xty`.
- Fix: read the right zones in place; a right zone that is the same handle as
  a left zone (`same_object`) reuses the left guard. Measure afterwards.

## `check()` holds a table entry per node

`check_points_agree` (`src/containers/field.rs`) allocates one `(zone, row)`
pair, 8 bytes, per point up to `point_bound` — 480 MB at 60 M nodes — and
walks every row of every zone, even when the zones share a single node. On the
bench field above it takes 0.16 s, which is also the check part of `u + g`.

Lead: a first pass marking each node "seen once / seen twice" in 2 bits (an
atomic bit set, as `sum_once` does), then a second pass comparing values only
at the nodes seen twice. To be measured: it trades memory for an extra pass.

## Writing into one zone can break the second coherence rule

Rule 2: a node carried by several zones holds the same value per common
component. `set_value`, `set_uniform` and `add_to_component` on `f[i]` touch
that zone only. Example: a field on `volume | face`, then
`T[1].set_uniform("T", 100.0)` — the face nodes hold 100 in the face zone and 0
in the volume zone. Only an explicit `check()` reports it.

- (i) Keep local writes, rely on `check()`: the error surfaces far from its
  cause.
- (ii) Write through the field — `f.set_value(node, c, v)`, `set_uniform` on a
  mesh — into every zone carrying the targeted nodes; a zone write stays
  possible, documented as local. Recommended.
- (iii) One value per node for the whole field, zones keeping only their node
  lists: divergence becomes impossible, but zones are shared between fields by
  `|`, solver outputs reuse the blocks' supports and reads are zero-copy — a
  redesign.

## A matrix block on a POI1 support repeating a node

A block numbers its rows by position in its POI1 support, and finds a node's
row through `SubMesh::node_index`, which ranks the **distinct** nodes: on a
support listing a node twice, the two disagree past the repeat.
`SubNodeField::from_poi1` refuses such a support (inverse map shorter than the
cells); the matrix constructors only document the requirement, and the test
`node_index_is_the_flat_position_only_on_a_deduplicated_support`
(`src/containers/matrix.rs`) pins it. `solve` on such a block now fails when
it builds its output field. Lead: the same refusal where a block takes its
support.

## `NodeField::with` and `with_components_per_submesh`

Rust `NodeField::with(mesh, components_per_submesh)`
(`src/containers/node_field.rs`) builds one zone per submesh, each with its own
component list; Python exposes it as `with_components_per_submesh`. One
function, two names, and the Rust one does not say what it builds. Pick one
name for both languages.

## `combine_columns` calls the operator through a function pointer

`combine_columns` (`src/containers/field.rs`) takes `op: fn(f64, f64) -> f64`
and calls it per value in its row loop; the pointer prevents inlining and
vectorisation. The loop is memory-bound today (first item), so measure before
changing it — e.g. a generic `F: Fn(f64, f64) -> f64` monomorphised per
operator, from the Python slots down to `merge_zones`.

## Node garbage collector

`Coords::gc()` (`src/coords.rs`) marks the nodes whose refcount fell to 0 as
dead. It frees no memory, ids are never reused, and pyrucast never calls it.
Python exposes it with `is_alive()` and `node_count()`, which only matter with
it (without gc, `node_count()` equals `capacity()`). The per-node refcount has
no other consumer than `gc`, `refcount(id)` and the `dump` column.

Decide: remove it from the API, or keep it undocumented. Mentions left either
way: `book/src/sauvegarde.md` (a reloaded `Coords` alone "qu'un `gc()`
collectera"), `book/src/aspect-informatique.md` ("`coords.gc()` collecte…"),
`book/src/memory-model.md` (per-node counter and mark-and-sweep paragraphs),
and the Rust docs of `src/coords.rs` and `src/atoms/node.rs`.

## Book pages not reviewed yet

Same rules as the reviewed pages: checked against the code, Python first then
"Côté Rust", current state only.

- `contraintes.md`.
- `operateurs/assemblage.md`: heading "Règle invariante : un Model = une
  Matrice".
- `operateurs/comportement.md`: heading `internal_forces(model, state)`; the
  signature is `internal_forces(model, state, solution, materials)`.
- `matrix.md`: describes a `DofId { node_id, field_idx }` type that does not
  exist; `factor` and `is_symmetric` are not cited.
- `evolution.md`.
- `elements/index.md`: the rest of the page.
- `sauvegarde.md`: a reloaded field is read through
  `o[...].coords().acquire(id)`; drop `gc()` and "la copie qu'un champ garde
  de la connectivité", which no longer exists.
- `compilation.md`: the features list lacks `spill`.

## Introduction diagram

`book/src/objets.md` draws the dependency diagram as a `<pre>` block with bold
names; `book/src/introduction.md` still has the plain `text` block of the same
diagram. Apply the same style there once its rendering on the site suits.

## "ne compose jamais" in the Aggregate page

`book/src/aggregate.md` says `+` "ne compose jamais". Between two fields on
different supports, `+` brings the zones of both sides into the result, a zone
present on one side only passing as is: it does gather zones. Reword: `|` is
the union of objects; `+` computes, and on fields it pairs zones by support.
