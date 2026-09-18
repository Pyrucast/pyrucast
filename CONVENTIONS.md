# Conventions — API, documentation and tests

This document settles the rules we no longer want to arbitrate case by case.

**First part, the API.** A single rule to decide whether an operation is an
inherent method of a container or a free function of the `ops/` modules, and a
single rule to project that decision onto the Python API. The goal: never again
decide "method or function?" by hand. The answer must fall out of the decision
tree below, and the Python form must be derived mechanically from the Rust form.

**Second part, documentation and tests** ("Documentation and tests", at the end
of the document). Which kind of example lives where, what each kind of test
proves, and **who checks it**. The narrative and the decision tables are in the
book, page [Documentation and tests](book/src/developper/documentation-et-tests.md);
only the rules appear here.

## Vocabulary

- **Container**: a type from `containers/` (`Mesh`, `SubMesh`, `NodeField`,
  `ElementField`, `Matrix`, `Model`, `FiniteElementSpace`, …). We call them
  *heavy* as opposed to scalars, component names, `NodeId`, `&[f64]` slices,
  etc.
- **Operator**: a function that consumes one or more containers and produces a
  new container (or a derived piece of data). Operators live in `ops/`, filed
  **by produced container** — `coords`, `element_field`, `export`, `field`,
  `geom`, `matrix`, `measure`, `mesh`, `node_field`, `solver`; see "Where a free
  function lives" below and `ops/mod.rs`.

## Rust rule: method vs free function

An operation is an **inherent method** if, and only if, all three conditions
hold:

1. there is **one** container that is the obvious `self`;
2. the operation *essentially* reads/writes that container — the other
   arguments are scalars, component names, `NodeId`s, `&[f64]` slices, …
   (never a second heavy container treated as a peer);
3. it is one of:
   - an **accessor** (`node_count`, `components`, `get`, …);
   - a **mutation that preserves the container's invariant** (`set`,
     `add_cell`, `add_to_component`, …);
   - a **cheap derived view** of that single container
     (`to_poi1_submesh` = the field's support seen as POI1).

Otherwise, it is a **free function in `ops/<theme>`**.

### Settling the borderline cases

> **Do two heavy containers come in as peers?**
> — Yes → free function (`ops::node_field::restrict(field, mesh)`).
> — No, it only reads `self` (+ small args) → method.

And a consistency landmark: if a single-container operation belongs to a
**family** already installed in `ops/` (the meshers, the assemblers), it joins
its family even if it could technically be a method. A family of operators is
never split between `ops/` and the container `impl`s.

### Accepted exceptions

- **Operator overloads** (`Add`, `Sub`, `Mul`, `Index`, …): always trait
  `impl`s on the container, never `ops::` functions. That is the idiomatic form
  in both languages. field+scalar and field+field arithmetic goes through them
  (`a + b`), not through an `ops::field::add`.
- **Named constructors** (`from_poi1`, `lagrange1`, `with_choices`, `block`):
  associated functions / `classmethod`s. They build *their own* type → they
  stay on the type. When that type is an **aggregate** (`Mesh`,
  `FiniteElementSpace`, `Model`, `ElementField`), the constructor lives at the
  **parent** level and returns a parent — see "Aggregates: one or several,
  transparently" below.
  **The limit of the exception: a catalogue is not a constructor.** A type that
  piles up dozens of them makes them visible on every one of its instances — in
  Python a `classmethod` can also be reached from an object, and
  `m.heat_conduction(fes)` runs while silently throwing `m` away. Past the
  plural, the family is filed like any other family of operators, in the module
  of the produced container: the physics — 22 Rust functions, 28 Python entries
  (the laws unfold) — are `ops::model::*` / `pyrucast.model.*`, not `Model::*`.
  A type keeps one or two named constructors, not a catalogue.

## Where a free function lives: the produced container

**A module of `ops/` gathers the operators that produce the same container, and
bears its name.** An operation is filed by its **output**, never by its input:
`gradient(field, fespace)` produces an `ElementField`, so it lives in
`ops::element_field`, next to `deformation` and `interp_to_gauss` — not in
`mesh` nor in `node_field`.

| module | produces |
|---|---|
| `ops::mesh` | a `Mesh` |
| `ops::node_field` | a `NodeField` |
| `ops::element_field` | an `ElementField` |
| `ops::matrix` | a `Matrix` |
| `ops::coords` | writes into the store |

Operators that produce **no** container escape the rule by construction; they
are filed by activity: `ops::measure` (reductions to a number), `ops::geom`
(geometric queries), `ops::export` (side effects).

A third case exists: the **generic** operator, whose product is a container —
always — but **not a determined container**. `abs` returns a `NodeField` or an
`ElementField` depending on what it is given: the rule therefore designates no
*single* module. Careful not to confuse this with two monomorphic functions of
the same family (`mask_nodes` / `mask_cells`), which each have a determined
product and are filed normally. These polymorphic operators are filed by
**domain**: `ops::field` (band masking, component filtering and renaming,
element-by-element maths). They remain full-fledged free functions, with their
method on each of the four field flavours.

### The single, named exception: `solver`

`ops::solver` produces a `NodeField` and should join `ops::node_field`. It keeps
its name because **several distinct families produce a nodal field** —
differentiation, assembly, solving — and only solving is looked up by its own
name. It is the only module named after an activity while producing a
container, and it must stay that way.

### Corollary: no qualifier in a function's name

A module **never** contains two operators that differ only by the container
they bear on. If that case arises, the qualifier belongs to the **module's
name**, not the function's name, and the module must be split. That is what
gives three homonymous, unambiguous merges — `mesh::consolidate`,
`node_field::consolidate`, `element_field::consolidate` — instead of three
suffixed functions in the same place. Likewise `coords::set` facing
`node_field::positions`, which reads.

### A limit of R3, worth knowing

R3 says that a function's qualifier must move into the module's name. The
remedy only applies if the qualifier distinguishes the **output**: that is the
case for the three `consolidate`s, which produce three different containers.

`select_nodes` and `select_cells` are out of reach: both produce a `Mesh` and
therefore both live in `ops::mesh`, their qualifier distinguishing the
**input**. It cannot be moved into the module's name, since the module is fixed
by the output. The suffix therefore remains legitimate, and it is on the Python
side that the ambiguity disappears — a single `select` function that dispatches
on the type it receives. Same situation for `integral` / `integral_element`.

### What is not built

Nothing in `ops/` produces a `FiniteElementSpace` or an `Evolution`, and that is
not an oversight: these containers are **declared** by a named constructor on
the type itself, they are not manufactured by transformation.

`Model` left that company on 2026-08-25. Its physics declarations form a
**catalogue** — 28 entries, and it grows with every physics added — which is a
module's job, not a type's surface: they live in `ops::model`
(`pyrucast.model.heat_conduction(fes)`), filed by product like everything else.
Declaring rather than computing is therefore not enough to stay on the type; the
family also has to fit in one or two names.

## The verb also exposed as a method

A free function keeps its **canonical form** — it is the one that is documented
and that defines the operation. It is **additionally** exposed as a method of
its first argument if, and only if, all **three** conditions hold:

1. **the first argument is the subject** — the object being transformed, not a
   parameter nor a support;
2. **the return is a container** — otherwise there is nothing to compose, and
   `f.integral(comp, fes) -> float` brings nothing over `integral(f, comp, fes)`;
3. **the operation makes sense for every instance of the type.** This is the
   condition that costs the most and is forgotten the fastest: a method
   *promises*, it appears in the autocompletion of every object of the type.
   `u.deformation(fes)` would appear on every nodal field although it requires
   `u_x`/`u_y`/`u_z` components; `t.thermal_strain(...)` on every element field
   although it requires a temperature. These operations remain free functions
   only.

The dividing line of condition 3: a **structural precondition** is admitted
(`triangulate_surface` wants a closed contour, `divergence` wants as many
components as axes — checked by counting, never by name), a requirement of
**meaning carried by component names** is not (`sigma_xx`, `u_x`, "a
temperature").

**How to test condition 3 without getting it wrong**: read the method with an
*arbitrary* receiver, not with the well-named example.
`stresses.internal_forces(model)` sounds right — but it is the variable's name
that does the work. `field.internal_forces()` reveals that the *type* promises
nothing: any element field would carry the method, although it requires the
Voigt stress. Compare with `field.sqrt()` or `field.mask(ge=0.0)`, which keep
their meaning on any field.

Two practical consequences:

- **An argument order that does not put the subject first is a defect to fix,
  not a reason to give up the method.** That is what moved
  `internal_forces(model, stresses)` to `(stresses, model)` and
  `solve_eliminate(model, matrix, rhs)` to `(matrix, model, rhs)`.
- **A symmetric operation has no method**: `a.merge(b)` would suggest that
  order matters. `merge` is the named alias of `a | b` — the operator already
  gives the symmetric form, and that is enough.

**The method does not re-document — but it must still *show* the
documentation.** On the Rust side, its `/// See [`mesh::skin`](fn@crate::ops::mesh::skin)`
is a link: rustdoc puts the reader one click from the full documentation. On the
Python side, the `.pyi` stub has no link mechanism at all, and a "See …" pointer
stays dead text there — that is all the IDE shows on hover.

**On the Python side, the method is therefore no longer written: it is
derived.** `#[py_op(method_on = PyMesh)]`, placed on the free function, extracts
the receiver from it, copies the following arguments, amputates the
`#[pyo3(signature = …)]` of its leading entry, and **copies the documentation as
literals** — it is that copy that makes it appear in full in `help()` as in the
stub. The attribute goes **above** the others, otherwise it would no longer see
the signature it has to rewrite, and takes `name = "…"` when the name changes
between the two forms (next section). It also inherits the function's
`#[allow(…)]`: a lint waiver that holds for it holds for its method, which
carries the same arguments.

Only two cases remain hand-written, and each for a reason that fits in one line:
a **receiver that is not a borrow** (`merge_nodes` returns the object itself,
hence `Py<Self>`), and a **method without a free function**, a canonical form in
its own right.

The **polymorphic operator** (`select`, `mask`) is no longer one of them. Its
method must not point back to the free function, which returns `Any`: the
receiver fixes the flavour, hence the produced type. The free function then
dispatches to **one function per flavour**, with a `PyRef<…>` subject and a
precise return, and it is on those that `#[py_op]` is placed. Each carries **its
own** documentation, written for that single receiver — a sub-field has no zones
to mention. These flavour functions are not registered in the module: the free
function remains the only entry point, and `#[pyfunction]` only serves there to
make valid the `#[pyo3(signature = …)]` that the attribute copies. Placed on a
`Bound` subject, the attribute refuses with these instructions, rather than
producing a method that would return `Any`.

### The name may change between the two forms

The full name is always "qualifier + verb"; what changes is where the qualifier
sits. The free function receives it from its **module**, the method has none and
must therefore **carry** it:

| free function | method |
|---|---|
| `matrix::stiffness(model, materials)` | `model.stiffness_matrix(materials)` |
| `matrix::mass(model, materials)` | `model.mass_matrix(materials)` |
| `matrix::tangent(...)` | `model.tangent_matrix(...)` |
| `matrix::geometric(...)` | `model.geometric_matrix(...)` |

When the output is of the subject's type, there is nothing to qualify and the
name does not move: `mesh::consolidate(m)` and `m.consolidate()`.

## Rust → Python rule: 1:1 mirror

- free function `ops::<theme>::f` → **top-level** Python function
  `pyrucast.f(...)`;
- Rust method `Type::m` → Python method `obj.m(...)`;
- Rust operator overload → Python dunder (`__add__`, `__getitem__`, …);
- Rust named constructor → Python `classmethod`.

No op is allowed to be a function on one side and a method on the other, nor to
change semantics between the two languages. The `py/` wrapper does not
**redesign** the API; it can only **restrict** it — see the exception below.

### Accepted exception: low-level Rust, curated Python

A single asymmetry is tolerated between the two surfaces: **Python may hide the
direct constructors of `Sub*` sub-objects** that Rust, for its part, exposes as
`pub`.

- **Rust side (low-level layer).** `SubMesh::new`, `SubElementField::new`,
  `SubMatrix::new`, `SubModel::heat_conduction` / `dirichlet`, … stay `pub`.
  The layer that writes the meshers, the assemblers and the parent constructors
  *must* be able to build `Sub*`s and place them behind a `Handle`; total
  control is the role of the Rust API.
- **Python side (curated surface).** `Sub*`s are **views** obtained by indexing
  the parent (`parent[i]`); they are **not constructed** directly. One builds at
  the parent level and composes with `|` (union, see "Aggregates: one or
  several" below). The unitary parent→sub coercion (`Aggregate::unit`) is the
  other face of that restriction: where an op needs a single sub-object, it is
  handed a unitary parent.

This is a **surface restriction**, not a redesign: Python invents no op, renames
none, changes no semantics; it simply **does not expose** certain constructors.
Everything exposed on both sides remains a 1:1 mirror. If a new op appears, it
follows the 1:1 rule by default; hiding a `Sub*` constructor is the **only**
deviation allowed, and it must stay limited to that case.

### The other direction: an op that Rust *cannot* carry

`pyrucast.mesh.from_gmsh` is the only free function that exists on the Python
side alone. It is not a surface choice: it reads the model of a live **gmsh**
session, so it requires a CPython interpreter carrying the `gmsh` module. Rust
cannot have it — a pure-Rust `cargo test` does not even link libpython.

The waiver stays narrow because the function **invents no operation**: it
fetches the current model's arrays and passes them to
`ops::mesh::from_gmsh_arrays`, the Rust operator, which is a strict mirror and
carries all the work. The criterion to remember for a future case: a
Python-only function is justified only if its *input* does not exist outside the
interpreter, and it must delegate its algorithm to a Rust operator. It then goes
into the `PYTHON_ONLY` of `tests/python/test_mirror_completeness.py`, with its
reason.

The style aimed for on the Python side is that of **numpy / scipy** (and the
**cast3m** heritage): **named** operators (`pyrucast.mesh.to_poi1(mesh)`,
`pyrucast.matrix.stiffness(model, mat)`) rather than method chains, and methods
reserved for accessors, mutations and derived views.

### The production module is reflected by a Python sub-module

Filing by produced container organises the Rust code (`src/ops/<module>/`)
**and** the Python API: a free function `ops::<module>::f` is exposed as
`pyrucast.<module>.f` (`pyrucast.mesh.to_poi1`,
`pyrucast.node_field.positions`, `pyrucast.matrix.stiffness`,
`pyrucast.solver.solve`, …). Containers (`containers::…`) and atoms
(`atoms::…`) remain top-level classes (`pyrucast.Coords`, `pyrucast.Mesh`,
`pyrucast.Node`, …). The mirror is **without exception**: no free function lives
at the Python top level.

The compiled `_pyrucast` extension, on the other hand, is **flat**: two
homonymous operators in two modules (the three `consolidate`s, `coords::set`)
carry a distinct `#[pyo3(name = "…")]` there, and the pure Python layer
re-exports them under their real name in the right sub-module. That is an
implementation detail of the private namespace, not a breach of the mirror.

It is the move to the maturin *mixed layout* (`python/pyrucast/` folder, private
`_pyrucast` extension + pure Python layer) that unlocks this filing: each module
is a real `.py` file re-exporting, in a typed way, the symbols of the flat
extension. A single `_pyrucast/__init__.pyi` stub is still generated for the
extension; since the sub-modules are only re-exports, the types follow them
without a dedicated stub.

### The wrapper files mirror the Rust tree

The Python namespace stays flat (above), but the **files** of the FFI layer
follow the same split as Rust — `containers/` (data) vs `ops/` (algorithms):

- **type wrappers** → `src/py/<type>.rs`, mirroring `src/containers/<type>`;
  only `#[pyclass]` + `#[pymethods]` live there (methods, views, dunders,
  `classmethod` constructors of the type);
- **operation wrappers** → `src/py/ops/<family>.rs`, mirroring
  `src/ops/<family>/`; only free `#[pyfunction]`s live there.

  **Exception, for `model` alone**: the wrapper of a physics of common shape is
  not written, it is **generated** by `physics_operator!` in the physics' file,
  from the same declaration as the Rust operator. `src/py/ops/model.rs` keeps
  only the shapes the macro does not cover. The reason is the cost of
  extension: a physics author writes physics, not plumbing — and a hand-written
  Python face is one more face to not forget.

This is a **navigation landmark**, not a change of surface: from a wrapper one
finds the Rust impl by path parity — `py/ops/mesh.rs` ↔ `ops/mesh/`, and `line`
↔ `ops/mesh/line.rs`. Corollary: a free function is filed by its **`ops`
family** (cf. the projection table and the explicitly settled cases below),
never by its input or output type.

## Aggregates: one or several, transparently

The containers `Mesh`, `FiniteElementSpace`, `Model` and `ElementField` are
**aggregates**: each is a `Vec<Handle<Sub>>` (see `aggregate.rs`). The purpose
of the aggregate is to handle **1 or several** sub-objects in one gesture,
transparently. For that to be *really* transparent in use, the user must never
have to build a sub-object and then attach it by hand, nor to "dive" into the
aggregate with `parent[0]` for the common case (a single sub-object).

Hence **a single rule**:

> **Named constructors live at the parent level and return a parent; parents are
> composed with `|` (union — Rust: `union`); the `Sub*` is an indexed view,
> never an object one builds-then-attaches.**

Three mechanical consequences:

1. **Building = a ready-to-use parent.** A named constructor that produces an
   aggregate returns the parent, not the sub-object. When it needs a support, it
   consumes the corresponding **parent** and sweeps its sub-objects: a unitary
   support → unitary aggregate, an N-zone support → N-zone aggregate. *That* is
   where the "1 or several" transparency lies. Precedents already in place:
   `FiniteElementSpace(mesh)` builds one sub-space per sub-mesh;
   `Mesh(config, element_type)` creates a mesh with one sub-mesh. Target:
   `model::heat_conduction(&fes)` creates one zone per sub-space.

2. **Composing = union (`|` in Python, `union` in Rust), never `add_sub` by
   hand.** To assemble heterogeneous physics / zones, one unions parents:
   Python `model.heat_conduction(fes) | model.dirichlet(...)`, Rust
   `model.union(&dirichlet)?`. The union clones the `Handle`s (refcount bump, no
   deep copy) and **deduplicates by handle**, so sub-objects are **shared**
   between parents. `add_sub` (one zone) and `add_subs` (all the zones of
   another aggregate, concatenated without deduplication) remain low-level
   primitives (and the internal path of the constructors), not the API for
   everyday use.

3. **The `Sub*` is a view, not a construction point (Python surface).** It is
   reached by indexing (`parent[i]`), exactly as `submesh[j] → Cell` and
   `cell[k] → Node` are already views. The `Sub*` keeps its own identity
   (`Handle` sharing, reference counting), but it leaves the construction path
   **on the Python side**: the `Sub*` constructors are not exposed there (that
   is the "Accepted exception: low-level Rust, curated Python" above). **On the
   Rust side**, `SubMesh::new` & co. stay `pub` — the low-level layer needs
   them.

   **Unitary case: `.unit()`, no re-coding at the parent.** To reach a method of
   the sub-object when the aggregate is unitary, we **do not expose** the `Sub`
   method on the parent (zero re-coding): we expose `.unit()` — the **view of
   the single sub-object**, a clear error if the aggregate is not exactly
   unitary — and we write `parent.unit().method(...)`
   (`mesh.unit().add_cell(...)`, `ef.unit().set_uniform(...)`,
   `K.unit().add_entry(...)`). It is more honest than `parent[0]` (which
   silently takes the first of several) and keeps the user aware that they are
   handling a unitary aggregate. Careful **not** to confuse it with the parent
   methods that *really aggregate* — sum/union/list/global (`Mesh::cell_count`,
   `Model::dual_vars`, `Matrix::n_rows`, …): those are not delegations and stay
   at the parent. On the Rust side, the low-level layer keeps its few
   delegations for convenience (`Mesh::add_cell`).

**Coercion at the boundaries.** Where an operation *really* requires a single
sub-object (e.g. the support of a `NodeField`, or `Matrix.block`), it accepts
the **parent** and unwraps its single sub-object via `Aggregate::unit` (explicit
error if the aggregate is not unitary). The user is never asked to supply the
`Sub*` itself.

What is **projected mechanically** onto Python (1:1 mirror): the parent's named
constructor (Rust `FiniteElementSpace::lagrange1` → Python `classmethod`), the
union (`union` → `__or__`), indexing → `__getitem__`. The only asymmetry is the
**non-exposure** of the `Sub*` constructors on the Python side (and the unitary
parent→sub coercion that accompanies it) — the exception described above. The
rule applies uniformly to the four aggregates and is meant to be carried by the
`impl_aggregate!` / `impl_aggregate_pymethods!` macros.

## A macro is justified by its number of expansions

Writing a macro costs more than writing the method it generates: it does not
read like Rust, its errors point into the expansion, and seeing the real code
requires `cargo expand`. That cost is fixed; what it buys grows with the number
of expansions. **Below a handful of expansions, we write the methods.**

The order of magnitude adopted is **four**. `impl_field_transform_pymethod!` is
expanded eleven times per family and amply pays for itself; `min`, `sum`,
`components`, `__pow__` or `__richcmp__`, generated once or twice, are
hand-written in their class's module.

**What a hand-written method shares goes into a function, not into a macro.**
The four `__richcmp__`s of the fields all call `py::field_slots::band_of`: the
semantics — which band of values a comparison states — lives in a single place,
in ordinary Rust, and is found by searching for its name.

**One macro per generated shape, and its name says the item produced.** Grouping
several shapes under the same name, each recognised by its keyword, shares
nothing: the rules of a `macro_rules!` share not one line, and the reader has to
learn a vocabulary in order to choose. Two container families are treated the
same way — `impl_field_transform_pymethod!` and
`impl_subfield_transform_pymethod!` rather than a family parameter, so that each
body names its trait and its access in plain sight.

The name follows three rules, in this order:

1. **It names what is produced, not what is passed in.** A macro binds *any*
   function of the expected shape; today's inventory is not a property of it.
   `impl_field_transform_pymethod!` and not `..._math!` — the generated method
   transforms a field into a field of the same flavour, whether the function
   passed in is `sqrt` or `normalize`. Same trap with arity: the bound function
   is unary, the produced method takes **no** argument.
2. **It names the Python species produced**, since the repository has a Rust
   side and a binding: `pymethod` for an ordinary method, `pyslot` for a slot,
   `pyfunction` for a free function. `mutator` alone would designate
   `Field::add_to_component` just as well, which is Rust.
3. **It carries the prefix of what it does to the declaration**: `impl_` when it
   adds methods to an already declared type, `define_` when it creates a fresh
   item (`define_polymorphic_pyfunction!` generates a function that did not
   exist).

Two to three words, as in the ecosystem (`vec!`, `bitflags!`,
`wrap_pyfunction!`): a macro name is re-read at every call, and `generate_` is
redundant with the `!` anyway.

## The direction of a macro: naming its types, or receiving them

A macro that serves several types can **name them itself** and loop over them,
or **receive them as an argument**, one call per type. The decision is not a
matter of taste: it follows the direction of the dependency.

**Downstream** — the macro lives in a module that already imports the types
served. It names them, and takes the **list** of those it serves:
`impl_field_mutator_pymethod! { /// … [PyNodeField, PyElementField], add_to_component }`,
the `///` doc at the head of the call. One call per family instead of four
calls, and above all the documentation written **only once** — one type per call
would have it copied as many times as there are flavours. That is the shape of
the macros in `src/py/field_macros.rs`.

**Upstream** — the macro lives in the module that defines the trait or the
machinery (`containers/field.rs`, `aggregate.rs`, `models/mod.rs`). It
**receives** its type, and each module calls it for what it implements. Having
it name `NodeField` or `PySubMesh` would invert the dependency — a container
does not know its implementers — and would take from each module the control of
what it implements.

**The exception of slots.** A slot (`__add__`, `__pow__`, `__richcmp__`,
`__len__`, `__repr__`, …) generates in pyo3 an `unsafe fn` trampoline that calls
another one. Edition 2024 no longer implicitly covers that body:
`unsafe_op_in_unsafe_fn` fires as soon as the `impl` lives outside the module
declaring the `#[pyclass]`. A slot macro therefore keeps the list shape, but is
**called from the `pyclass`'s module**, with a list of a single type. A slot
whose stub declares the CPython names by hand (`__ge__`/`__gt__`/`__le__`
/`__lt__` for `__richcmp__`) must furthermore **not** be decorated with
`gen_stub_pymethods`, otherwise it would be counted twice in the `.pyi`.

## Three display levels

Every object exposes three display levels, in layers, all wired to Python. Each
level has a distinct role and **never spills over** into the next:

| Level | Rust | Python | Role | Bound |
|---|---|---|---|---|
| summary | `Display` | `__str__` | one line: identity + key dimensions | O(1), never any content |
| structure | `Debug` | `__repr__` | counters, dimensions, names, handles, metadata (`{:#?}` indented) | bounded, never bulk content |
| content | `dump::Dump` | `dump(precision=3, max_rows=20, max_cols=12)` | full content: matrix grids, value tables, connectivity | bounded by `DumpOptions` (elision `… (N more)`) |

Rules:

- **Gradation**: each level says **at least** what the previous one says. A
  `repr` poorer than its `str`, or a `dump` that omits a metadata item of the
  `repr`, inverts the hierarchy and is fixed. It is held **by hand**, field by
  field — no test guards it, that is the price of idiomatic `Debug`s
  (`debug_struct`, `debug_map`) rather than ones built in layers.
- **Identity is that of the handle, and it is the aggregate that shows it.** A
  sub-object does not know the handle that carries it; only its holder can. An
  aggregate therefore names its zones at all three levels — `[#7f3a2c, …]` in
  the summary (elided beyond three, to stay bounded), `{<SubMesh #7f3a2c>: …}`
  in the structure, `── [0] <SubMesh #7f3a2c> ──` in the content. A sub-object
  displayed alone has no identifier, in Rust as in Python.
- **Summary and structure do not lock.** The `Display` of a `Handle`
  deliberately forbids itself from doing so — it may be formatted while a write
  guard is held, and reading would cause a deadlock — and the same caution holds
  for the views (`Cell`, `Element`): their `Display` and `Debug` show only the
  carrying handle and the index. The element type, the connectivity and the
  positions require a guard, so they live in `dump`, called knowingly. `{:?}` is
  written in error messages, sometimes while holding the very lock at fault.
- `Display`/`Debug` **never** dump bulk content (values, connectivity, grid). A
  `repr` stays bounded whatever the size of the object.
- `dump()` **prints directly to the terminal** and returns nothing (`()` in
  Rust, `None` in Python). On the Python side the printing goes through Python's
  `print` (respects `sys.stdout`, redirections, test capture). The core
  `Dump::render(&self, opts) -> String` produces the string (composition of
  aggregates); it is not exposed to Python.
- Matrices are dumped as a **labelled dense grid**: the `(node, var)` DOFs label
  rows and columns directly.
- Generic aggregates (`Mesh`, `FiniteElementSpace`, `Model`, `ElementField`)
  dump the summary then the indented `dump` of each sub-object; `Matrix` dumps a
  single global grid.

On the implementation side: trait + shared helpers in `src/dump.rs` (each type
implements `render`, `dump` is provided by default); macro
`impl_aggregate_dump!` for the generic aggregates; `impl_dump_pymethod!` for the
Python wiring of the non-aggregate wrappers.

## Projection table (target state)

On the Python side, functions live in the **sub-module of the produced
container** (`pyrucast.<module>.f(...)`) — see the note on sub-modules above,
without exception.

| Operation | Rust | Python |
|---|---|---|
| single-container accessor / mutation | method | method |
| derived view of a single container | method | method |
| mesh→mesh transformation | `ops::mesh::*` | `pyrucast.mesh.to_poi1`, `pyrucast.mesh.consolidate`, … |
| production of a nodal field | `ops::node_field::*` | `pyrucast.node_field.positions`, `pyrucast.node_field.restrict`, `pyrucast.node_field.merge` |
| production of an element field | `ops::element_field::*` | `pyrucast.element_field.gradient`, `pyrucast.element_field.material_field` |
| reduction to a number | `ops::measure::*` | `pyrucast.measure.integral`, `pyrucast.measure.xty` |
| writing into the store | `ops::coords::*` | `pyrucast.coords.set`, `pyrucast.coords.displace` |
| field → **same** field (polymorphic) | `ops::field::*` | `pyrucast.field.mask`, `pyrucast.field.sqrt`, `pyrucast.field.filter_components` |
| writing an external format | `ops::export::*` | `pyrucast.export.export_vtk` |
| declaration of a physics | `ops::model::*` | `pyrucast.model.heat_conduction`, `pyrucast.model.dirichlet` |
| assembly `Model` → `Matrix` | `ops::matrix::*` | `pyrucast.matrix.stiffness`, `pyrucast.matrix.mass` |
| solving `A·x = b` | `ops::solver::*` | `pyrucast.solver.solve` |
| arithmetic (`+ - * /`, indexing) | operator `impl` | dunder |
| named constructor | associated fn | `classmethod` |
| verb eligible under the three conditions | free function **and** method | same — see "The verb also exposed as a method" |

`ops::geom` does not appear: its two functions (`locate_points`,
`project_points`) are the internal primitives of `model.embedded` and
`model.contact`, and are not exposed to Python. It is the only whole-module
waiver, recorded in `tests/python/test_mirror_completeness.py`.

## Explicitly settled cases

- `restrict(field, mesh)` → **`ops::node_field`** (field + mesh as peers;
  produces a nodal field).
- `merge(a, b)` → **`ops::node_field`** (two fields as peers); named alias of
  the union `a | b` (`Aggregate::union`), a non-arithmetic merge.
- field+field addition → **operator `+`** (arithmetic of values, not the
  composition of zones), not an `ops::field::add`. The `+ - * /` operators (zone
  to zone **and** aggregate to aggregate) combine **by `(support, component)`**
  in union/passthrough (a component or support on one side only = raw value
  unchanged); the operands need neither the same set of components nor the same
  decomposition. Primitives: `SubField::merge_components` (zone) /
  `Field::merge_field` (aggregate), `Field::merge_subfield` (targeted update of
  one zone). Where a component mismatch must be an error (`Evolution`
  interpolation), `SubField::check_same_components` guards `merge_components`
  upstream.
- `stiffness(model, mat)`, `mass(model)` → **`ops::matrix`** (assembler family;
  `mass` follows `stiffness`, they do not get separated).
- `consolidate(mesh)`, `to_poi1(mesh)` → **`ops::mesh`** (single-container, but
  they produce a `Mesh` and belong to the mesher family).
- `select(field, ge=…)` → **`ops::mesh`**: it starts from a field but returns a
  `Mesh`, and one files by the output. Its twin `mask`, which rewrites the
  values without changing the structure, returns a field of the kind received
  and therefore stays in `ops::field`.
- `material_field(model, …)` → **`ops::element_field`** (produces an
  `ElementField`). The old `build` module, which designated no family,
  disappears.
- `flux`, `internal_forces` → **`ops::node_field`**: these are assemblies, but
  their result is a vector, not an operator. The machinery they share with
  `ops::matrix` (`ops::coloring`, `ops::scatter`) lives at the root of `ops` —
  those are not operators.
- `mul_dense(self, x: &[f64])` → **method** (`x` is a slice, not a heavy
  container: a single-container matrix-vector product).
- `support_submesh` / `support_mesh` on `NodeField` → **methods** (view of the
  support of a single field). Renamed from `to_poi1_submesh` / `to_poi1_mesh` so
  as not to be confused with the operator `ops::mesh::to_poi1(mesh)`.
- `coords()` is reserved for the **return to the container**: on `Mesh`,
  `SubMesh`, `NodeField`, `Matrix` and `Node`, it returns the `Coords` carried.
  The *values* are therefore called **position** everywhere: `position()` /
  `set_position(…)` on `Node` as on `Coords`, and `node_field::positions(mesh)`
  for the field that reads them all. The old names — `coord()` on a node,
  `coordinates` for the operator — put on the same object two argument-less
  methods that nothing told apart (`mesh.coords()` facing
  `mesh.coordinates()`), one returning the container, the other the values.

---

# Documentation and tests

The narrative, the decision tables and the why are in the book, page
**[Documentation and tests](book/src/developper/documentation-et-tests.md)**.
Below, the rules alone.

## The four rules

1. **No page of the book owns any code, without exception.** Every ` ```rust `
   or ` ```python ` fence on a page contains a `{{#include}}` pointing at a
   source that CI runs. An example copied by hand into a page is code that
   nothing checks; it rots silently.

   **What is not code is tagged ` ```text `**: a signature annotated with
   `/* … */`, an enumeration abridged by `// … one line per physics`, a
   pseudo-code `f(...)`. This is not a waiver, it is an observation — syntax
   highlighting would lie, and the reader must see straight away that they
   cannot copy that block. There is therefore no longer a "sketch page": the
   nature is judged **block by block**, never page by page.

   An often forgotten corollary: a Rust declaration copied into a page
   (`pub struct Handle<T>`, `pub trait Cancel`) **is code**. It exists in
   `src/`, it gets anchored and it gets included.

2. **Every public API item carries an executable example** in its documentation
   (`///`). That is point 3 of the *Definition of Done*. `ignore` is
   **forbidden** in a doctest: it is the marker that checks nothing, and that is
   exactly what disarmed the book 73 times. When the example cannot run,
   `no_run` (it compiles, it does not execute) or `compile_fail` (we document
   what the type system forbids). The setup lines are hidden with `# `, they
   stay compiled.

3. **An example lives where it is executed**, never copied. Where, depending on
   what it illustrates:

   | the example illustrates… | it lives… | the book shows it via… |
   |---|---|---|
   | a Rust API item | a doctest on the item | nothing — it is in the rustdoc |
   | a complete Rust chain | `tests/<subject>.rs`, anchored | `{{#include ../../tests/<subject>.rs:anchor}}` |
   | a complete Python chain | `examples/<subject>.py` | `{{#include}}`, whole or anchored |
   | a teaching walkthrough | `formation/<subject>.py`, anchored | `{{#include …:anchor}}` |
   | the use of an operator in Python | `tests/python/test_doc_<family>.py`, anchored | `{{#include …:anchor}}` |
   | what is **not code** | the page itself | ` ```text `, without highlighting |

   **A pure-delegation method carries no example.** It has no logic at all: it
   calls the free function, receiver included. The latter is the canonical form,
   it is the one that is documented — demanding an example of them would
   duplicate its own, and would give a second text to maintain for nothing.

   They are recognised by their **marker**: their entire documentation fits in a
   "see [`module::verb`]". It is that marker that counts, not the location —
   these blocks live in `src/ops/**/methods.rs`, but also at the bottom of
   `src/ops/matrix.rs`, and some are born of a macro on `impl $T`. The ratchet
   excludes them from the denominator on that criterion.

   **The Python surface answers to the same rule, at a looser granularity.** Its
   docstrings are written in the `///` of `src/py/`, and the module is compiled:
   one doctest per item would cost a home-made collector there, since the
   standard library's `DocTestFinder` does not descend into the functions of an
   extension module. What is required instead is that a public entry be **cited**
   by an executed example of the book — a `tests/python/test_doc_*.py`, read by
   AST so that a name in a comment or in a string does not count.

   The guarantee is narrower than that of the Rust ratchet, and that must be
   said: the granularity is the *name*, not the call, so that a `.get(` cited
   once holds for the `get`s of every container. It suffices for what is asked
   of it — **no public entry is absent from the examples**, and a new entry
   cannot arrive without being shown. A class counts as cited as soon as one of
   its methods is: idiomatic Python writes `mesh.unit()`, never `SubMesh(...)`.

   The `test_` prefix is not decorative: `pytest` only collects `test_*.py`. A
   file of example sources named otherwise is included in the book and
   **executed by nobody** — exactly what the rule seeks to prevent.

   **Anchored code lives at module level, not inside a test function.** mdbook
   does not strip the indentation of an included excerpt: anchored inside a
   function, it would show up shifted by four spaces, which no user would write.
   At module level the file also reads in the order of the page, the variables
   flowing from one anchor to the next.

   Two consequences. pytest executes the file at **collection**: an example that
   breaks is a collection error — full traceback and non-zero return code —
   hence `--continue-on-collection-errors` in `pyproject.toml`. And the
   **fixtures are not available**: an excerpt that writes files under a short
   name moves itself into a temporary folder, and **gives the current directory
   back at the end** of the module. Omitting the restitution would displace all
   the test files collected afterwards.

   Corollary: **a doctest and a book block are not the same thing.** The doctest
   serves the rustdoc reader — API reference, item by item; the book block
   serves the chapter's reader — narrative, complete chain. Two audiences, two
   sources, and we do not try to unify them.

4. **A guard is only installed once it has been deliberately broken**, at least
   once, **return code checked** — not just the display. A verification step
   that goes green without looking at anything costs more than its absence: it
   reads as coverage. Three cases measured in this repository as of 2026-08-18:
   `mdbook test` tests zero blocks; a non-existent `{{#include}}` anchor returns
   an **empty** block with a return code of **0** and no message; a missing
   included file only produces an `[ERROR]` in the log, return code **0** as
   well.

## What each test proves

| kind | where | what it proves |
|---|---|---|
| unit test | `#[cfg(test)] mod tests` in `src/**.rs` | the behaviour of a unit, public or private |
| doctest | `///` and `//!` in `src/**.rs` | that the example documenting an item compiles and runs |
| integration test | `tests/*.rs` | a complete chain seen from outside the crate |
| Python test | `tests/python/*.py` | the pyo3 surface and the behaviour on the Python side |
| guard | `tests/python/test_method_exposure.py`, `test_mirror_completeness.py` | an **API invariant**, not a behaviour |
| example | `examples/*.py` | an end-to-end user chain |
| training script | `formation/*.py` | a complete teaching walkthrough |
| bench | `benches/*.rs` | a **performance**, never a correctness |

A bench never replaces a test: it measures, it asserts nothing. It does not run
in CI.

## Choice of inclusion mechanism

`{{#include}}` if cargo (or pytest, or `run_examples`) already compiles the
file — the verification happens at the source, mdbook only displays it.
`{{#rustdoc_include}}` is only of interest for a file that *nothing else*
compiles: it passes the whole file to rustdoc while displaying only the anchor.
There is none in this repository today, and there is no reason to create one.

## A waiver carries a reason

Every sketch page, every item without an example, every exclusion from a guard
lives in a **name → reason dictionary**, accompanied by a hygiene test that
fails if the entry becomes stale. That is the pattern already in place in
`tests/python/test_method_exposure.py` and `test_mirror_completeness.py`; no
other is created.
