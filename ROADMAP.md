# pyrucast — Roadmap

This document says **where the project stands** and **what could come next**. The
first part is a dated status report; the second is a list of **unsettled** leads
— they will be refined, ordered or dropped later.

Status as of **5 August 2026** (v0.2.1).

## Philosophy

- Finite element library: Rust core + Python API, inspired by the principles of cast3m.
- **Simple, maintainable code, editable by a non-expert human**.
- External dependencies kept to the strict minimum — **explicit agreement required before any addition**.

## Locked architecture decisions

| Topic | Decision |
|---|---|
| Memory | Objects live behind a `Handle<T>` — a one-field wrapper over `Arc<RwLock<T>>`: reference counting by the `Arc`, one lock per object, owned guards. No global registry, no `Session` object to pass around; the handle *is* the object's address. |
| Layer separation | `containers/` (structures) ⊥ `ops/` (operators) ⊥ `py/` (binding). A module of `ops/` bears the name of the **container it produces**. The full rules are in `CONVENTIONS.md`. |
| Geometric primitives | `nalgebra` (small vectors and matrices — geometry, meshing, visualisation), `nalgebra-sparse` for sparse storage. |
| Linear algebra (solver) | **Direct sparse LU from `faer`**, multithreaded, with a factorisation cache on the `Matrix` (*factorise once, solve often*). `SolveMethod` is the extension point for another back-end. |
| Serialisation | `serde` + `bincode` through a **single** `Portable` trait, the byte contract of file save/reload. |
| Parallelism | `rayon`, **always on**, carried *above* the physics kernels: a kernel sees neither rayon, nor a handle, nor a lock. |
| Python binding | `pyo3` + `maturin`, *mixed layout*: private flat `_pyrucast` extension + pure Python layer that files it into sub-modules. |
| Documentation | `mdbook` (theory + doctests) + rustdoc, published on GitHub Pages. |
| Non-linear / transient algorithms | **Orchestrated in Python**, not in Rust (see below). |
| Method | Breadth first: all the structures + bindings + doc/tests before the heavy numerics. |

Three decisions were **revised along the way**, and are settled here for good:
the solver is not a home-made implementation behind a `LinearSolver` trait (it is
`faer`); there is no `Session` object to pass around; and the central store
itself was removed — slab, generations, home-made counter and disk swap all
duplicated what the `Arc` already does, and the swap freed nothing. See
[book/src/memory-model.md](book/src/memory-model.md).

### Approved dependencies (frozen base)

Always linked: `serde`, `bincode`, `nalgebra`, `nalgebra-sparse`, `faer` (the solver's sparse LU), `rayon` (parallelism), `parking_lot` (the objects' locks), `paste` (aggregate macros). Optional, behind a feature: `pyo3`, `pyo3-stub-gen`, and the visualisation ones `plotters`, `winit`, `softbuffer`. Tooling: `maturin`, `mdbook`, `ruff`, `criterion`. Any other addition = a new explicit request.

### Definition of Done per object

1. Rust struct addressable by `Handle<T>`
2. `Debug` (structural) + `Display` (cast3m-style listing summary)
3. Rust unit tests + doctests on the whole public API
4. PyO3 binding: `__repr__` → `Debug`, `__str__` → `Display`
5. Python tests (pytest)
6. mdbook chapter (theory + API)

An object is only finished once these 6 points are green.

---

# What is done and available

In figures: **82,600 lines of Rust**, 1,046 Rust unit tests (+ 20 integration
test files), 447 Python tests, 27 doctests, 75 book pages, 26 examples, 6
training scripts, **100 functions exposed in Python**.

## Memory base

`Handle<T>` = `Arc<RwLock<T>>`: counting by the `Arc`, infallible `read`/`write`,
identity through `same_object`. **Two-level** refcount — the objects on one side,
the nodes of a `Coords` on the other, with `gc()`. One lock per object, owned
guards allowing reading **in place**. `compact()` gives back the tail memory.

## Containers and atoms

The **seven aggregates** — `Mesh`, `FiniteElementSpace`, `NodeField`,
`ElementField`, `Model`, `Matrix`, `Evolution` — each with its `Sub*` view,
`len` / `[i]` / `|`. Plus `Coords` (the coordinate store, multiple sets,
axisymmetric frame) and the atoms: `Node`, `Cell`, `Element`, `ElementType`,
`Point*` / `Vector*`, `Band`, `RgbColor`.

## Finite elements — 16 types

`POI1`; linear `SEG2`, `TRI3`, `QUA4`, `TET4`, `PYRA5`, `PENTA6`, `HEX8`
(Lagrange-1); quadratic `SEG3`, `TRI6`, `QUA8`, `QUA9`, `TET10`, `PENTA15`,
`HEX20`, `HEX27` (Lagrange-2, serendipity or complete). Shape functions and
derivatives, Jacobian including the *manifold* case, `Gauss` and `Reduced`
quadratures — plus the pyramid's conical Gauss × Jacobi quadrature. Axisymmetry
carried by `Coords` and integrated into the single `det_j_w` measure.

Each type fits in **one file** `atoms/element_kind/<name>.rs` implementing the
`ElementKind` trait — reference nodes, facets, edges, domain, interpolation,
quadrature, VTK/gmsh codes, families. A single `match`,
`ElementType::as_kind()`, connects the enum to the behaviour, on the model of
`SubModel::as_kind()` on the physics side: adding an element costs one file and
two variants, and no generic consumer changes.

## Physics — 16 sub-models

Thermal: `HeatConduction`, `BoundaryTransfer` (surface exchange with an ambient
medium — thermal film, mass transfer, or elastic foundation depending on the
components it is given),
`Radiation` (radiation to infinity `σε(T⁴ − T_∞⁴)`: stiffness linearised around
`T_∞`, exact residual, consistent tangent validated by finite differences; the
first physics to declare **two** natures, `[Thermal, Radiation]`).
Diffusion: `Fick` (concentration `c` / flux `j`, its own `Diffusion` nature),
`InterfaceTransfer` (exchange `h(a₁ − a₂)` between two coincident but separately
numbered meshes; like `BoundaryTransfer`, it takes the quantities it transfers as
an argument — thermal, diffusion, or a bonded joint of finite stiffness — and
both share their kernel in `models::transfer`).
Mechanics: `Truss`, `Elasticity` (plane stress/strain, axisymmetric, 3-D),
`Plasticity` (the **flow law as an attribute**: perfect von Mises or with
isotropic hardening, non-associated Drucker-Prager with apex handling,
four-parameter Ottosen integrated by secant plane; then the **time-dependent**
laws — Norton, Lemaitre and Blackburn creeps, Chaboche viscoplasticity and its
damaging Lemaitre-Chaboche variant, which error out in the absence of `dt`;
tangents all confronted with a finite difference of the internal forces),
`Damage` (**law as an attribute** too: scalar Mazars, two-variable Damage TC —
the only one that gives its stiffness back to a crack that closes again — and
orthotropic SiC/SiC, whose damage directions are the weaving frame reused from
elastic orthotropy), `Timoshenko`,
`Frame` (2-D frame), `Frame3d`, `Bernoulli` (beam without transverse shear, 1-D
/ plane / spatial, exact at the nodes through Hermite interpolation),
`Shell` (**formulation as an attribute**: Reissner-Mindlin with under-integrated
shear against locking, or discrete Kirchhoff DKT/DKQ which has no shear to lock;
six DOF per node, drilling tied to the membrane rotation, membrane and drilling
shared by both). Constraints: `Dirichlet`, `Mpc`, `Embedded` (immersion),
`Contact` (node-surface, unilateral).
Uncoupled thermal expansion (`thermal_strain`, `alpha` as an optional material
component — accepted by elasticity, plasticity and damage: the expansion is
subtracted before the mechanical law sees anything at all, so nothing forbade a
material that expands from yielding).

**Material symmetry** (`MaterialSymmetry`, `src/models/symmetry.rs`): an axis
orthogonal to the kinematic assumption, shared by `Elasticity`,
`HeatConduction` and `Fick` — isotropic (default, unchanged), orthotropic,
anisotropic. The orthotropy frame is given by **vectors** carried by the material
field (`V1X/V1Y`, plus `V1Z` and `V2*` in 3-D), like Cast3M's
`MATE 'DIRECTION' V1 V2`. The rotation of the elasticity tensor goes through
order 4 rather than through a Bond matrix, which removes any index convention;
isotropy short-circuits that path and keeps its exact numbers.

The cost of adding a physics is **O(1) file**: a struct + an
`impl SubModelKind`, two lines of wiring.

`FollowerPressure` (a load whose direction turns with the surface, built on the
deformed tangents and not on Nanson — `I + ∇_s u` is not a deformation gradient
on a manifold) is **temporarily removed**: the only physics without a matrix, it
borrowed a `stiffness_layout` to declare its integration geometry. Code and
re-integration points in
[`archive/pression-suiveuse.md`](archive/pression-suiveuse.md); to be put back
with a "load" capacity distinct from `Domain`.

## Assembly

Three forms of contribution (`Contribution`): `Computed` (integrated on the fly
and scattered into the CSR), `Literal` (values already filled in — Dirichlet,
MPC) and `Coupling` (an **inter-mesh** block, rows on one mesh and columns on
another; its scatter is sequential, since a colouring over a single connectivity
no longer proves disjointness).

Four matrix kinds behind a single machinery (`MatrixKind`): stiffness /
conductivity, mass / capacity, geometric stiffness, consistent tangent — plus
`lump`. Sparsity pattern memoised **per kind** on the `Model`, element matrices
computed in parallel and scattered into the CSR by **cell colouring**, without
materialising any COO. Multi-quadrature elements (Timoshenko) on the same path.
Internal forces `∫ Bᵀσ` (Cast3M `BSIG`) and divergence through the same nodal
scatter driver.

## Solver

Direct sparse LU (`faer`), factorisation cached on the `Matrix`. Three routes:
**Lagrange** (`solve`, augmented system), **elimination / condensation**
(`solve_eliminate`), **unilateral active-set** (`solve_unilateral`). Output on
block supports (reused POI1 handles, subtractable fields).

## Meshing

Primitives: `line`, `circle`, `arc`, `transfinite`, `points`.
Sweeps: `sweep`, `extrude` (TRI3 → PENTA6, QUA4 → HEX8), `revolve`,
`sweep_solid`.
Free meshers: `triangulate_surface` (constrained Delaunay + Ruppert, holes
included, TRI3/QUA4, 2-D and planar 3-D contours) and `triangulate_volume`
(exact predicates, frozen hull, recovery, refinement and anti-*sliver*
smoothing).
Advancing-front meshers: `pave_surface` (quadrangles) and `pave_volume` (HEX8
boundary layer + PYRA5 junction + TET4 core).
Topology and transformations: `skin`, `border`, `orient`, `consolidate`,
`merge_nodes`, `to_quadratic`, `convert`, `to_poi1`, `translate`, `rotate`,
symmetries, geometric selections (sphere, plane, cylinder, cone, torus, line).
Input/output: **gmsh** reading (MSH 2.2 and 4.1, ASCII and binary), **in-memory
gmsh** import from a live session (no file, no copy of the arrays), **VTK**
export.

## Fields and operators

Kinematics (`gradient`, `deformation`, `beam_deformation`,
`frame_deformation`), behaviour (`behavior`), materials (`material_field`,
`interp_to_gauss`), positions, `restrict`, `flux`, `divergence`,
`internal_forces`, field masks and arithmetic, reductions (`integral`, `xtx`,
`xty`), geometric queries (`locate_points`, `project_points`, `contact_gaps`),
`Evolution` (tabulated interpolated value). 168 free functions in `ops/`.

## Parallelism

`rayon` always on, centralised grain policy (`parallel::MIN_PARALLEL_LEN`).
`models::kernel` drivers above pure, sequential kernels; zero-copy through
guards held for the whole parallel region. **Bit-for-bit** determinism for the
write-once operators and the reductions; determinism through colouring (not
bit-for-bit) for the assembly and the nodal scatters; the solver is the
exception (faer back-end).

## Python API

*Mixed layout*: private flat `_pyrucast` extension, filed by the pure Python
layer into sub-modules named after the produced container (`pyrucast.mesh`,
`pyrucast.element_field`, `pyrucast.matrix`, …), the containers and atoms
staying at the top level. Versioned `.pyi` stub.
Cooperative interruption (`Ctrl+C`) through the `Cancel` trait.
Non-linear orchestration in pure Python: `pyrucast.thermomechanics`
(step-by-step thermal→mechanical), modified Newton accelerated by Anderson in
the examples.

## Visualisation

CPU rendering with `plotters` (PNG/SVG) and an interactive `winit`/`softbuffer`
window: meshes, fields coloured **per element** (never averaged between
elements), interpolated rendering by subdivision, curves, axes, gizmo.

## Tooling

`script/check_all.sh` (the full pass, to be wired into CI) and the five blocks it
chains (`check_format`, `check_rust`, `check_python`, `check_examples`,
`check_doc`), runnable in isolation; `build.sh` / `dev.sh`, `run_examples.sh`,
`set_new_version.sh`, `scaling.sh`. Each has its PowerShell equivalent. GitHub
Actions CI: publication of the book and the rustdoc on Pages, multi-OS release
(Linux, Windows, macOS).

---

# Future leads

Nothing that follows is settled: neither the order, nor the scope, nor even the
fact of doing it. These points are the ones the status report showed up as
**missing**, to be refined later.

## Transient analysis and time integration

The most visible functional gap. The mass and the *lumping* exist; what is
missing is the **driving**: a time integration scheme (θ-method, Newmark), and
the Python layer that orchestrates it — the equivalent of a `PASAPAS`. Related to
it is **advection** (`ADVE`), the only Cast3M brick still absent on the matrix
side.

## Solver and performance

- **Renumbering** — `Coords` already carries an optional permutation separating
  the solver order from the identity, but nobody computes it: a bandwidth/profile
  reduction (Cuthill–McKee style) remains to be written, with the invariance of
  the results as a test.
- **Iterative methods** — `SolveMethod` is the extension point, already carrying
  `Lu` and `Cholesky`. A conjugate gradient would be the first one to need a
  preconditioner, hence a matrix-matrix product.
- **A performance pass** on a large mesh, with the benches (`benches/parallel.rs`,
  `benches/geom.rs`, `script/scaling.sh`) as the instrument.
- **A global allocator** — see below.

### Global allocator: taking back the page faults of the large fields

**The problem.** An operator that produces a field returns a **fresh** container.
On a serious mesh, that container is enormous: `behavior::integrate` on 3.61 M
QUA4 in axisymmetry returns 462 MB (4 Gauss points × 4 components × 8 bytes).
Past its threshold, glibc serves such a block by `mmap` and gives it back by
`munmap` at `Drop`. The memory therefore returns to the kernel at every call, and
the next call **refaults it page by page**: 112,812 pages of 4 KB, one at a time,
once per call.

Measured (`perf stat`, 1900×1900 grid): **11.9 M page faults** for about a
hundred calls — exactly the theoretical count, so neither a leak nor waste,
simply the price of allocating a fresh half-gigabyte. Each fault costs
**1.07 µs** (kernel entry, allocation of a page, zeroing of 4 KB, table update),
that is **120 ms per call, ~15% of an 819 ms call**. The effective bandwidth
falls to 560 MB/s: the operation is limited not by memory throughput but by the
latency of putting the pages back into service.

This does not only concern the benches: a Newton loop or a transient calls
`integrate` again at every iteration, and therefore pays at every iteration.

**The solution.** Replace the global allocator with an arena allocator
(`mimalloc` or `jemalloc`), in one line in `lib.rs`:

```rust
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

**Why it helps.** An arena allocator **keeps** the freed blocks instead of
returning them to the kernel, and recycles them on the next round. The pages
therefore stay mapped: we only fault them on the first call. Verified by raising
`MALLOC_MMAP_THRESHOLD_`, which produces the same effect on glibc — **11.9 M →
402 k faults, that is 30×**, at a rigorously constant instruction count (3.506 G
vs 3.511 G). That really is kernel time removed, not computation moved around.

The benefit goes beyond `integrate`: the assembly and the matrices allocate much
bigger, and would benefit from the same recycling. It is also portable to macOS
and Windows, where the equivalent `mallopt` setting would only hold for glibc.

**What it costs.** One more dependency in the base, and a higher resident
footprint since the arenas are kept. To be measured before/after on `benches/`
**and** on the RSS, not only on the time.

**The deeper alternative.** An `integrate_into(&mut field, …)` form reusing the
field of the previous round would remove the allocation itself, not only its
faults. But it widens the `Domain::integrate_behavior` seam, whereas the contract
wants a physics author to write only `integrate_point` — a design project in its
own right.

## Save and restore

**Done.** `save` / `load` write an object graph under file-local identifiers and
read it back preserving the sharing — two fields carried by the same support
remain, after reading back, two fields carried by a single support. Named roots,
a dictionary on the way out as on the way in, reading back adds without replacing
anything, the counters are recounted from zero, the header is versioned and an
unknown version is refused. See
[book/src/sauvegarde.md](book/src/sauvegarde.md).

Two possible follow-ups, neither committed:

- **partial reading** — extracting one object from a very large file without
  loading everything. It costs an index at the head of the file (identifier →
  position), at the price of the single forward loop that makes reading back
  simple. The format version leaves the door open.
- **an interchange export** (HDF5 or another), not to be confused with this
  format: HDF5 has no notion of object identity nor of shared reference, the hard
  part would have to be rewritten on top of it, and it would cost a C dependency
  in the base. It will justify itself on its own merits — publishing results —
  and as a separate operator.

## Mesh quality

The most advanced area of the code, and the least planned. The open point is
documented in `triangulate_volume.rs`: the **edge recovery** of the hull is
unfinished — a blocked edge remains dependent on flips, failing that on
`allow_surface_nodes`. Added to that are the quality measures and their tracking
over time.

On the **quadrangle** side, the state of the art, the place `pave_surface`,
`grid_surface` and `merge_triangles` hold in it, and six quantified leads are in
[MAILLAGE-QUADRANGULAIRE.md](MAILLAGE-QUADRANGULAIRE.md) — with the list of what
was tried and then set aside, not to be done again.

## Memory evolutions (conditional)

The model is deliberately minimal: one `Arc<RwLock<T>>` per object, one counter
per node in `Coords`. Two extensions are identified, each **triggered by a
measurement**, not by anticipation.

1. **Enumerating the live objects** — a cast3m-style listing, or a save of a
   whole session. Costs a `Vec<Weak<_>>` per type registered in `Handle::new`,
   the single creation funnel that exists for that purpose.
   *Trigger*: a user command that needs it.
2. **Simplifying the `Node` contract** — a pure cast3m mode, where only a mesh
   keeps a node alive. Removes at a stroke the per-node counter *and* the
   cancellation logic of `add_cell`. *Trigger*: observing that isolated `Node`s
   are of little practical use. Details in
   [book/src/memory-model.md](book/src/memory-model.md).

---

# Appendix — non-linear and transient algorithms: Python orchestration

The non-linear schemes (**Newton** loop) and the **time integration** are **not**
coded in Rust: they are **driven on the Python side**, by functions that compose
the core's operators.

This is the **Cast3M** model, where these algorithms are not native operators but
**GIBIANE procedures** (`PASAPAS`, `UNPAS`, `TRANSNON`…) chaining the basic
operators (`RIGI`, `KTAN`, `BSIG`, `RESO`, `COMP`, `EXCO`…). Here, the
orchestration language is **Python** instead of GIBIANE:

- **the Rust core supplies the bricks** — assembly (`stiffness`, `mass`,
  `geometric`, `tangent`), linear solving (`solve`, `solve_eliminate`,
  `solve_unilateral`), behaviour integration (`behavior`), internal forces
  (`internal_forces`), field operations;
- **Python assembles the algorithm** — the Newton loop (residual, tangent,
  increment, convergence test) and the time scheme are Python functions calling
  those bricks.

That is already the case in practice: `pyrucast.thermomechanics` runs a
step-by-step thermal→mechanical scheme, and the examples run a modified Newton
accelerated by Anderson. An equivalent of `pasapas` / `unpas` / `transnon` will
therefore be a **Python library** shipped with the binding, not a Rust operator
(cf. the "Équivalent pyrucast" column of `opérateur_castem.csv`).
