# Quadrilateral meshing — state of the art and leads

A working note, **unsettled**. It says where pyrucast's quadrilateral meshers
stand in the literature, what separates them from the state of the art, and the
six leads identified — with, for each, what has been measured.

Status as of **24 August 2026**. A companion to [ROADMAP.md](ROADMAP.md),
section *Mesh quality*.

**Lead 7** was opened and closed on 24 August: it is implemented, and it is the
only point of this note that is no longer a lead.

---

## 1. Where this note comes from

A crenellated box with thirteen plates, meshed by `pave_surface` and by
`grid_surface`, served as a case study. It brought out four defects, all fixed:

| commit | defect |
|---|---|
| `6bf4f6f` | a flat neighbourhood grid allocated 4 GiB and killed the process |
| `e53785a` | the seam of a flat ring left a crack, opened into a hole by the smoothing |
| `41594a9` | a seam did not carry the triangles along, and could lay a chord on the contour |
| `3c78950` | the LIFO stack of the loops crushed the first row of a contour |
| `946677c`, `51afb9f` | two cleanup gestures: pairs of flat triangles, and a node shared by one triangle and two quadrangles |

After which, on that case: **worst cell 0.451**, 3 cells below 0.5 out of
10,120, zero holes. What remains is structural, hence this note.

---

## 2. The five lineages of the literature

### 2.1 Direct paving — the lineage of `pave_surface`

Blacker & Stephenson, *Paving: a new approach to automated quadrilateral mesh
generation*, IJNME **32**:811–847 (1991). A front that advances in rows, with
seaming, unsticking and closure.

Still being worked on: *[An improved Q-Morph algorithm for quad-dominant hybrid
mesh generation with advanced front propagation and topology
optimization](https://link.springer.com/article/10.1007/s00366-025-02196-y)*,
*Engineering with Computers* **41**:4255–4275 (2025), enriches the front types
for concavities and adds a topological optimisation through **predefined
templates, cavity remeshing and elimination of triangle pairs** — that is,
exactly the gestures added in `946677c` and `51afb9f`.

### 2.2 Indirect advancing front — Q-Morph

Owen, Staten, Canann & Saigal, *[Q-Morph: an indirect approach to advancing
front quad
meshing](https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-0207(19990330)44:9%3C1317::AID-NME532%3E3.0.CO;2-N)*,
IJNME **44**:1317–1340 (1999). Triangulate first, then transform the triangles
in an order dictated by a front.

**Superseded** by the next lineage as early as 2012: not to be written today.

### 2.3 Indirect optimal — Blossom

Remacle, Lambrechts, Seny, Marchandise, Johnen & Geuzaine, *[Blossom-Quad: a
non-uniform quadrilateral mesh generator using a minimum-cost perfect-matching
algorithm](https://onlinelibrary.wiley.com/doi/10.1002/nme.3279)*, IJNME
**89**:1102–1119 (2012).

The graph has one vertex per triangle, one edge per adjacent pair, weighted by
the quality of the quadrangle the pair would make. On it one solves the
**minimum-cost perfect matching** (Edmonds' algorithm), in polynomial time and
**exactly**: no local pass can do better, and if the triangle count is even and a
perfect matching exists, no triangle is left.

Completed by the triangular mesher made for it: Remacle et al., *[A frontal
Delaunay quad mesh generator using the L∞
norm](https://onlinelibrary.wiley.com/doi/10.1002/nme.4458)*, IJNME (2013), which
produces **almost right-angled** triangles — the L∞ norm is what makes the pairs
give squares.

### 2.4 Direction fields and parametrisation — the dominant line

One does not build cells, one builds a **field**, and the mesh falls out of it. A
cross field (directions with order-4 symmetry) carries **singularities** that are
exactly the future irregular vertices, and Poincaré–Hopf imposes the sum of their
indices: **the number of irregulars is not negotiable, only their position is.**

Mandatory entry points — the two surveys:

- Bommes, Lévy, Pietroni, Puppo, Silva, Tarini & Zorin, *[Quad-Mesh Generation
  and Processing: A Survey](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.12014)*,
  CGF **32**:51–76 (2013);
- Campen, *Partitioning Surfaces into Quadrilateral Patches: A Survey*, CGF
  **36**:567–588 (2017).

Milestones: QuadCover (Kälberer et al., 2007), *Mixed-Integer Quadrangulation*
(Bommes et al., 2009), *[Globally optimal direction
fields](https://dl.acm.org/doi/10.1145/2461912.2462005)* (Knöppel et al., 2013),
*[Integrable PolyVector fields](https://dl.acm.org/doi/10.1145/2766906)*
(Diamanti et al., 2015).

**Quantization** — going from a continuous field to a mesh with integer edges:
*Quantized global parametrization* (Campen et al., 2015), *Quad layouts via
constrained T-mesh quantization* (Lyon et al., 2021), *Min-deviation-flow in
bi-directed graphs for T-mesh quantization* (Heistermann et al., 2023). And the
theory: *Which cross fields can be quadrangulated?* (Shen et al., 2022) — not all
fields can be.

The fast, pragmatic side: *[Instant Field-Aligned
Meshes](https://dl.acm.org/doi/10.1145/2816795.2818078)* (Jakob, Tarini, Panozzo
& Sorkine-Hornung, SIGGRAPH Asia 2015) — joint local smoothing of an orientation
field and a position field, without global optimisation, hence linear and
interactive.

The most recent and the most relevant here: Couplet, Chemin, Bommes & Chien,
*[Surface Quadrilateral Meshing from Integrable Odeco
Fields](https://arxiv.org/abs/2604.03889)* and *Size-controlled quadrilateral
meshing using integrable odeco fields*, SGP 2026 — the **size map** done
properly, through integrable frame fields with alignment **and size**
constraints.

### 2.5 The grid school — the lineage of `grid_surface`

Liang & Zhang, *[Guaranteed-quality all-quadrilateral mesh generation with
feature
preservation](https://www.sciencedirect.com/science/article/abs/pii/S0045782510000836)*,
CMAME (2010), and *Hexagon-based all-quadrilateral mesh generation with
guaranteed angle bounds*, CMAME (2011).

A quadtree governed by curvature, **2-refinement templates without hanging
nodes**, then a two-layer **buffer zone** created by removing the elements near
the boundary. Hard guarantee: **all angles within [45°, 135°]**.

That is structurally `grid_surface` (grid core + front band), with two
differences: they grade by quadtree — which `grid_surface` tried and then
dropped — and above all they **prove** an angle bound instead of measuring it.
More recent: *[Boundary constrained quadrilateral mesh generation based on domain
decomposition and
templates](https://www.sciencedirect.com/science/article/abs/pii/S004579492400004X)*,
*Computers & Structures* (2024).

### 2.6 The neural wave (2024–2026) — to be placed

*Learning Direction Fields for Quad Mesh Generation* (Dielen et al., 2021),
*[NeurCross](https://dl.acm.org/doi/10.1145/3731159)* (ACM TOG 2025), then the
autoregressive ones — *[QuadGPT](https://arxiv.org/html/2509.21420v1)*, QuadLink,
TopGen (2026). They aim at "game-ready" retopology, **not at computation**: no
guarantee of validity nor of exact respect of the contour. A living
bibliography: [quad-meshing-survey](https://github.com/Bigger-and-Stronger/quad-meshing-survey).

---

## 3. The map: pyrucast ↔ literature

| what pyrucast does | the corresponding state of the art | the gap |
|---|---|---|
| `pave_surface` | Paving 1991, + improved Q-Morph 2025 | **up to date on that branch**: the three gestures of the 2025 paper are those of `946677c` / `51afb9f` |
| `grid_surface`, `grid_surface2` | Liang & Zhang 2010–2011 | same architecture (core + buffer); the **angle guarantee** missing |
| `merge_triangles` (greedy: merging + grouping) | Blossom-Quad 2012 | **one generation behind on an identical problem** |
| `cleanup` (doublets, valences, three-cell stars) | CleanUp 1997; QuadQS: cavities + patterns guided by the singularities | **up to date on the basic gesture** (lead 7); it remains that we repair blindly, while they know **where** an irregular is allowed to be |
| size: one scalar per domain | integrable odeco, SGP 2026 | absent |
| — | cross field, quantization, layout | absent, and it is the backbone of the rest |

Gmsh is the synthesis of all of it: Reberol, Georgiadis & Geuzaine,
*[Quasi-structured quadrilateral meshing in
Gmsh](https://arxiv.org/abs/2103.04652)* (2021) — cross field + size map →
frontal insertion → Blossom recombination → **midpoint subdivision**
(all-quadrangle guaranteed) → topological remeshing keeping the irregulars that
correspond to a singularity of the field, each operation undone if the quality
drops.

---

## 4. The six leads

### Lead 1 — Optimal matching in `merge_triangles`

Replace the greedy pass (merging, then grouping into a hexagon) with
Blossom-Quad's **minimum-cost perfect matching**.

- *Measured*: on a crenellated frame, `triangulate_surface` in QUA4 leaves
  **147 triangles out of 588 cells**, worst cell 0.000 (0.331 after `regularize`
  + `cleanup`). An exact matching would leave zero or one.
- *Reservation*: the reference implementation (Blossom V, Kolmogorov) is under a
  research licence, **incompatible with MPL-2.0**. Edmonds has to be written, or
  a permissive crate found.
- *Verdict*: **the best gain/effort ratio of the list.** A known problem, an
  exact published solution, a scope bounded to one operator.

### Lead 2 — The parity discipline, outside `all_quad`

The two places that decide on a split gap already impose "both halves stay even",
but **only under `all_quad`**:

```rust
// unstick
if gap < 3 || n - gap < 3 || (all_quad && gap.is_multiple_of(2)) { continue; }
// find_seam
if all_quad && gap % 2 == 1 { continue; }
```

Yet the 65 triangles of the box **all** come from the closure of a small ring (49
from a 3-node ring), and a ring's parity is decided in the **splits**, not in the
rows.

- *Measured*, with the discipline enabled alone (without requiring zero
  triangles):

  | | current | parity on |
  |---|---:|---:|
  | box, pave `round` `along` | 65 tri, worst 0.451, 5th pct. 0.551 | **27** tri, 0.423, 0.451 |
  | box, pave `round` `none` | 133 tri, 0.257 | **75**, **0.343** |
  | box, `grid_surface` | 65 tri, 5th pct. 0.476 | **9** tri, 5th pct. **0.550** |

- *Open point*: it costs +1,263 cells on the `round` + `along` configuration and
  brings the 5th percentile there back down from 0.551 to 0.451. To understand
  why there and not elsewhere.
- *Verdict*: **one day, the figures are already there.**

### Lead 3 — Midpoint subdivision

An unconditional all-quadrangle safety net: each triangle gives three
quadrangles, each quadrangle four. ×4 cells, size halved, zero triangles, **and
never a contour refusal**. That is what gmsh does in QuadQS before the
topological remeshing.

- *Direct motivation*: `all_quad=True` today **refuses** a contour of odd parity
  ("the outer boundary loop has 1151 segments — an odd number"). The box of the
  study is therefore not entitled to it.
- *Verdict*: a few dozen lines, no risk, but a density trade-off that the caller
  must choose.

### Lead 4 — A size map

The target size is **one scalar per domain**. The state-of-the-art meshers accept
a size field with a gradient limit; `grid_surface` does take its lines from the
contour, but the caller cannot dictate a variable density.

- *Verdict*: **the only point of the list that users would see directly.** About
  one week for a scalar version interpolated on a background mesh; the "clean"
  version requires lead 6.

### Lead 5 — Landing on a live front (`aim_at_live`)

`aim_at_frozen` already shortens the advance in order to **lay** a row on a grid
core. Nothing equivalent exists between two live fronts: they collide.

- *Verdict*: **the interest has melted away.** It was the answer to the crushing
  of the front, settled otherwise by `3c78950` and `51afb9f` — only 3 cells below
  0.5 out of 10,120 remain. To keep in reserve, not as a priority.

### Lead 6 — The cross field

The real leap, and the condition of the others: it says **where** an irregular
vertex is allowed to be, it carries the size map, and it opens the layout.

- *Verdict*: **a project**, and the only one of the list that demands theory as
  much as code. It is what would separate pyrucast from the state of the art
  rather than from the state of the art of 1991.

### Lead 7 — Collapsing the poor stars — **DONE**

An interior node that has only **three** cells around it can be given up, and one
cell with it. Kinney, *[CleanUp: Improving Quadrilateral Finite Element
Meshes](https://people.eecs.berkeley.edu/~jrs/meshpapers/Kinney.pdf)*, 4th IMR
(1997), case `3-4+34+000`: "*all three quads around the center node are deleted
and a fill_2 is used to fill the hole. Four irregular nodes are replaced with
zero irregular nodes.*"

`cleanup` had **one** of the four cases, written as a special case ("pentagon",
two quadrangles and one triangle). The identity that unifies them: around a node
carrying \( q \) quadrangles and \( t \) triangles, each quadrangle lays two
edges that do not touch it and each triangle one, so the star is bordered by a
polygon with \( n = 2q + t \) sides; and a decomposition of an \( n \)-gon
without an interior node satisfies \( 2q' + t' = n - 2 \). With \( q + t = 3 \),
the re-split always exists, and always with one cell fewer.

| \( q, t \) | border | before | after | cells |
|---|---|---|---|---|
| 3, 0 | hexagon | 3 quadrangles | 2 quadrangles | 3 → 2 |
| 2, 1 | pentagon | 2 quadrangles, 1 triangle | 1 of each | 3 → 2 |
| 1, 2 | quadrangle | 1 quadrangle, 2 triangles | 1 quadrangle | 3 → 1 |
| 0, 3 | triangle | 3 triangles | 1 triangle | 3 → 1 |

**The point that cost three attempts**: the gesture cannot be judged on the spot.
It *removes* a node, so the remaining ring is mechanically stretched until
something releases it — which always happens, since the pavers smooth after every
row. Measured on the spot, the re-split almost always looks worse than the star
it replaces: the quality floor of `switch_diagonals` (70%), applied as is,
removed **53 useful gestures out of 61**. Yet `switch_diagonals` can afford it —
it moves no node, so what it measures is final.

The order adopted, which is gmsh's: apply, **relax the ring**, measure, undo
entirely if the worst cell of the neighbourhood has dropped. Three details proved
decisive there, each through a measurement:

- the relaxation must be **kept** with the gesture. Judging on positions that are
  then rejected is measuring a mesh that nobody receives;
- the trial relaxation must carry **the same guard as the real smoother** (no
  step that flips a cell). A bare Laplacian walks, near a concave corner, towards
  a point the monotone smoother will never reach, and makes the gesture accepted
  on a promise that will not be kept: the house fell to **0.055** of worst cell;
- a **pre-filter** guards the entry: nothing is attempted if the gesture brings
  neither valence nor shape. The verdict after relaxation judges the
  *neighbourhood*, so locally, and does not see that a mediocre cell elsewhere has
  just become the worst of the mesh. Without it one gains thirteen irregulars and
  loses the non-regression guarantee on the worst cell — a bad trade, a worst
  cell that goes backwards breaks a computation.

**The 3-3 pair, added afterwards.** Two interior nodes of valence 3 joined by an
edge are out of reach of the gesture above: giving up only one trades one
irregular for two, and the pre-filter refuses it. Together they carry only
**four** quadrangles — their stars overlap on the two cells of the shared edge —
bordered by a **hexagon** in all twenty-seven cases found on the box, without
exception. Two nodes and two cells go at once, for a valence gain of +2, up to +4
when the ring carries a 5. Examined **before** the lone node, failing which the
latter takes one of the two and the pair never gets its chance.

Extended afterwards to the **3-4** pairs, whose star carries five cells bordered
by a **heptagon**, re-split into two quadrangles and one triangle — the one that
was already there, parity forbidding the creation of one.

| | `grid_surface` | `pave_surface` |
|---|---:|---:|
| irregulars | 185 → 164 → **104** | 739 → **703** → 708 |
| valence error | 194 → 164 → **104** | 754 → **718** → 714 |
| interior valence 3 | 58 → 46 → **16** | 329 → **311** → 312 |
| 1st percentile | 0.706 → 0.714 → **0.824** | 0.625 → 0.633 → **0.641** |
| worst cell | 0.461 → 0.461 → 0.437 | 0.141 → **0.284** → 0.284 |

*(columns: before the pair, 3-3 pair, then 3-4.)* On `grid_surface`, two 3-3
pairs detected bring down twelve valence-3 nodes and not four: one collapse
unblocks others in cascade. The 3-4 case removes thirty more, at the price of
0.019 on the worst cell — an accepted trade-off, the first percentile gaining
0.11 in the same move.

*Measured*, raw output of the pavers, before → after:

| | cells | worst | 1st pct. | 5th pct. | irregulars | valence error |
|---|---:|---:|---:|---:|---:|---:|
| box, `grid_surface` | 11,749 → **11,523** | 0.456 → **0.461** | 0.572 → **0.706** | 0.760 → **0.844** | 568 → **185** | 724 → **194** |
| circle, `grid_surface` | 1,260 → **1,236** | 0.288 → **0.366** | | | | |
| house, `grid_surface` | 470 → **460** | 0.420 → 0.420 | | | | |
| rounded square, `grid` | 441 → **424** | 0.244 → **0.340** | | | | |
| rounded square, `grid2` | 415 → **409** | 0.400 → 0.371 | | | | |

On the box, the **274** interior valence-3 nodes were *all* surrounded by three
quadrangles — the only case `cleanup` did not know how to handle. 58 remain, and
that is the floor: Poincaré–Hopf fixes the number of irregulars, only their
position is negotiable (§ 2.4).

The only regression is the rounded square in `grid_surface2`, 0.400 → 0.371.

**An independent bug, found along the way.** The quality measures are signed, and
a cell read clockwise counts as negative — which reads as *flipped*. Yet an
entirely clockwise mesh is in no way abnormal: a paver returns the orientation of
the contour it was given, so that a domain meshed from a reversed outer contour
comes out clockwise. Read as is, **the three `improve` operators silently refused
to touch it**: `regularize` moved no node (maximum displacement measured: `0.0`),
`cleanup` found nothing, and `merge_triangles` left the box's 74 triangles at 74
— against 64 once the mesh was flipped. Fixed in `Surface::read`, which
normalises the orientation on reading and restores it on output.

---

## 5. What was tried and set aside

Not to be done again without a new reason. All these attempts were measured then
removed from the repository.

| attempt | result |
|---|---|
| Abandoning the row as soon as **one** node falls below the relaxation floor (thresholds 0.5 / 0.35 / 0.25 / 0.15) | a cell with Jacobian **0.000** appears at every threshold, including on the circle in grid mode |
| End-of-row demotion of a node that can no longer advance | a net gain on the angular shapes (20×20 square: 600 → 544 cells, worst 0.541 → 0.778) but **house 0.491 → 0.147** and **circle in grid mode 5th pct. 0.796 → 0.608** |
| Full **FIFO** traversal of the loops | your box gains, but **`grid_surface` + `along`: worst 0.312 → 0.047** |
| FIFO traversal **for all** the loops, including under a grid core | `grid_surface` on a plate with a round hole **no longer terminates** (> 4 min against 0.05 s) |
| Forcing an even front at **every** row | 65 triangles → **685**, worst cell 0.000 |
| Same, restricted to the last rows before closure (thresholds 8 / 12 / 20 / 40) | no gain; at 12, `grid_surface` makes holes again |
| An **immediate** quality floor on the collapse of a star (70%, that of `switch_diagonals`) | 53 useful gestures out of 61 removed. The gesture removes a node: it can only be judged after relaxation (lead 7) |
| Trial relaxation **given back** after the verdict | measures a mesh that nobody receives: rounded square `grid2` 0.468 in the trial, 0.331 delivered |
| Trial relaxation as a **bare Laplacian**, without a validity guard | promises a position the monotone smoother does not reach: house `grid_surface` worst cell **0.055** |
| Collapse **without a pre-filter**: the verdict after relaxation as the only condition | gains on everything we were aiming at — box 185 → **162** irregulars, 58 → **45** valence-3 nodes, 11,523 → 11,506 cells, and `pave_surface` benefits too (house 620 → 595) — but the worst cell **goes back below the starting point**: box 0.461 → 0.436 for 0.456 at the start, house `grid_surface` 0.420 → **0.346**. The verdict is local and does not see that a mediocre cell elsewhere has become the worst of the mesh. Making the verdict global is another project |
| Refusing the seam that would lay a chord on the contour | it does settle the holes, but on a crenellated strip the front folds back and **the call fails** — losing the mesh to avoid a zero-area crack is a bad trade. Replaced by the re-seaming of `41594a9` |

One methodological point that has served several times: **front relaxation and
traversal order gain on the angular shapes and lose on the curved ones.** Any
global setting of these two levers is paid for somewhere; that is why the
relaxation became a choice of the caller (`relax`) rather than an imposed
setting.
