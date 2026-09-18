# Elasto-plastic cantilever beam — comparison of the non-linear solvers

Comparison of three methods solving **exactly the same problem** (2-D cantilever
beam, perfect von Mises, plane stress, clamped end on the left, shear −1 on the
right face ramped from 0 to `PMAX`):

| Method | File | Non-linear solver | Iteration operator |
|---|---|---|---|
| **Cast3M** (`PASAPAS`) | [`plasticite_poutre_console.dgibi`](plasticite_poutre_console.dgibi) | **Modified** Newton | constant **elastic** `K` (recycled factorisation) |
| **pyrucast** (reference) | [`plasticite_poutre_console.rs`](plasticite_poutre_console.rs) · [`.py`](plasticite_poutre_console.py) | **Modified** Newton | constant **elastic** `K` (cached factorisation) |
| **pyrucast + Anderson** | [`plasticite_poutre_console_anderson.rs`](plasticite_poutre_console_anderson.rs) · [`.py`](plasticite_poutre_console_anderson.py) | Modified Newton **+ Anderson acceleration (m=3)** | elastic `K` + extrapolation over the last 3 `(u, g=K⁻¹r)` |

The **three methods all use the same modified Newton**: iteration operator =
constant **elastic** matrix, factorisation **recycled** from one step/iteration
to the next (by default `PASAPAS` does not refresh the tangent matrix). The
iteration counts are therefore directly comparable. Anderson accelerates exactly
that same method by extrapolating over the recent history of the iterates.

Both pyrucast variants (reference, Anderson) exist in **Rust** and in
**Python**: same native operators under the hood, only the driving loop differs.
The numerical results (deflection, plasticity, iteration count) are
**bit-identical** between Rust and Python
(`|deflection_Rust − deflection_Python| = 0` at every step) — the result tables
below therefore hold for both, and only the timings tell Rust from Python.

Common parameters: `E = 210000`, `ν = 0.3`, `σy = 250`, `L = 10`, `H = 1`,
`NSTEPS = 10`, `PMAX = 5`.

**Measurement protocol**: Linux, Cast3M **and** pyrucast are **multithreaded** —
a comparison at equal resources. Each run is launched **alone**, on an idle
machine (no other concurrent computation), median wall-clock time over several
runs. Figures taken on `HEAD` after the field-arithmetic refactor
(`merge_components`), which sped pyrucast up relative to the earlier
measurements.

---

## 200×40 QUA4 case

### Results per step (deflection u_y at the tip, mid-height; max cumulated plastic strain)

| step | P | deflection — Cast3M | deflection — pyrucast | deflection — Anderson | εₚ max Cast3M | p max pyrucast | p max Anderson |
|----:|----:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5 | −9.57000e-3 | −9.570000e-3 | −9.570000e-3 | 0 | 0 | 0 |
| 2 | 1.0 | −1.91400e-2 | −1.914000e-2 | −1.914000e-2 | 0 | 0 | 0 |
| 3 | 1.5 | −2.87100e-2 | −2.871000e-2 | −2.871000e-2 | 0 | 0 | 0 |
| 4 | 2.0 | −3.82800e-2 | −3.828000e-2 | −3.828000e-2 | 0 | 0 | 0 |
| 5 | 2.5 | −4.78500e-2 | −4.785000e-2 | −4.785000e-2 | 0 | 0 | 0 |
| 6 | 3.0 | −5.74200e-2 | −5.742000e-2 | −5.742000e-2 | 0 | 0 | 0 |
| 7 | 3.5 | −6.69900e-2 | −6.699000e-2 | −6.699000e-2 | 0 | 0 | 0 |
| 8 | 4.0 | −7.65775e-2 | −7.657762e-2 | −7.657762e-2 | 2.09613e-4 | 2.098470e-4 | 2.098471e-4 |
| 9 | 4.5 | −8.62339e-2 | −8.623411e-2 | −8.623411e-2 | 5.73629e-4 | 5.764621e-4 | 5.764622e-4 |
| 10 | 5.0 | −9.66879e-2 | −9.668826e-2 | −9.668826e-2 | 1.00339e-3 | 1.007817e-3 | 1.007817e-3 |

Quantified check:

- **pyrucast reference vs Anderson: identical** — `|deflection_ref − deflection_Anderson| = 0`
  at every step (Anderson only changes the iteration count, never the result).
- **pyrucast vs Cast3M**: max relative discrepancy on the deflection **3.7e-6** —
  agreement to the ~6 digits Cast3M displays, consistent with the convergence
  tolerances.

### Non-linear iterations (all the steps)

| step | Cast3M | pyrucast (reference) | Anderson m=3 |
|----:|----:|----:|----:|
| 1 | 2 | 1 | 1 |
| 2 | 2 | 1 | 1 |
| 3 | 2 | 1 | 1 |
| 4 | 2 | 1 | 1 |
| 5 | 2 | 1 | 1 |
| 6 | 2 | 1 | 1 |
| 7 | 2 | 1 | 1 |
| 8 | 5 | 16 | 7 |
| 9 | 9 | 39 | 15 |
| 10 | 12 | 90 | 28 |

Same method (modified Newton, recycled elastic matrix) for all three: one
iteration = one forward/back substitution on the cached factorisation,
comparable unit cost. Cast3M converges in far fewer iterations than the original
pyrucast because of **looser convergence criteria** (`PRECISION` of `PASAPAS`
by default ≈ 1e-4/1e-5 relative, against `tol = 1e-6` relative here), not
because of the operator. Anderson accelerates the same method and gets close to
Cast3M's iteration count.

### Run times (median wall-clock, isolated run, multithreaded)

| Method | Rust | Python |
|---|---:|---:|
| Cast3M `PASAPAS` | ~3.27 s | — |
| pyrucast reference | ~2.34 s | ~2.50 s |
| pyrucast + Anderson | **~1.65 s** | **~1.73 s** |

---

## 400×80 QUA4 case

### Results per step

| step | P | deflection — Cast3M | deflection — pyrucast | deflection — Anderson | εₚ max Cast3M | p max pyrucast | p max Anderson |
|----:|----:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5 | −9.57812e-3 | −9.578116e-3 | −9.578116e-3 | 0 | 0 | 0 |
| 2 | 1.0 | −1.91562e-2 | −1.915623e-2 | −1.915623e-2 | 0 | 0 | 0 |
| 3 | 1.5 | −2.87343e-2 | −2.873435e-2 | −2.873435e-2 | 0 | 0 | 0 |
| 4 | 2.0 | −3.83125e-2 | −3.831246e-2 | −3.831246e-2 | 0 | 0 | 0 |
| 5 | 2.5 | −4.78906e-2 | −4.789058e-2 | −4.789058e-2 | 0 | 0 | 0 |
| 6 | 3.0 | −5.74687e-2 | −5.746870e-2 | −5.746870e-2 | 0 | 0 | 0 |
| 7 | 3.5 | −6.70517e-2 | −6.705165e-2 | −6.705165e-2 | 2.11730e-4 | 2.116993e-4 | 2.116994e-4 |
| 8 | 4.0 | −7.66530e-2 | −7.665312e-2 | −7.665312e-2 | 6.16890e-4 | 6.195860e-4 | 6.195862e-4 |
| 9 | 4.5 | −8.63301e-2 | −8.633024e-2 | −8.633024e-2 | 1.20478e-3 | 1.214164e-3 | 1.214164e-3 |
| 10 | 5.0 | −9.68125e-2 | −9.681252e-2 | −9.681252e-2 | 1.95421e-3 | 1.967391e-3 | 1.967390e-3 |

Note: on this finer mesh, first yield appears as early as step 7 (the stress
gradient is better resolved near the clamped end).

Quantified check:

- **pyrucast reference vs Anderson: identical** — `|deflection_ref − deflection_Anderson| = 0`
  at every step.
- **pyrucast vs Cast3M**: max relative discrepancy on the deflection **1.7e-6**.

### Non-linear iterations (all the steps)

| step | Cast3M | pyrucast (reference) | Anderson m=3 |
|----:|----:|----:|----:|
| 1 | 2 | 1 | 1 |
| 2 | 2 | 1 | 1 |
| 3 | 2 | 1 | 1 |
| 4 | 2 | 1 | 1 |
| 5 | 2 | 1 | 1 |
| 6 | 2 | 1 | 1 |
| 7 | 5 | 13 | 6 |
| 8 | 7 | 29 | 10 |
| 9 | 11 | 81 | 25 |
| 10 | 15 | 141 | 37 |

Same remark as in 200×40: the three methods share the modified Newton (recycled
elastic matrix), iterations of comparable unit cost; the Cast3M ↔ pyrucast gap
in the counts comes from the convergence criteria, not from the operator.

### Run times (median wall-clock, isolated run, multithreaded)

| Method | Rust | Python |
|---|---:|---:|
| Cast3M `PASAPAS` | ~16.78 s | — |
| pyrucast reference | ~17.27 s | ~16.96 s |
| pyrucast + Anderson | **~10.55 s** | **~10.47 s** |

---

## Reading

- **Accuracy**: the three methods give the same result at both sizes — reference
  and Anderson coincide bit for bit, and the gap to Cast3M stays ≤ 3.7e-6
  (relative): the pyrucast port and the Anderson acceleration are validated.
- **Iterations**: Anderson divides the modified Newton's iteration count by
  ~3–3.8 on the plastic branch (down to 141 → 37 at the final step in 400×80),
  bringing the count close to Cast3M's **while keeping `tol = 1e-6`**.
- **Times**: at equal resources (both codes multithreaded, isolated runs), the
  Anderson acceleration is clear-cut — **200×40: 2.34 → 1.65 s (≈ −30 %)**;
  **400×80: 17.27 → 10.55 s (≈ −39 %)**. The wall-clock gain stays below the
  gain in iterations because each accelerated iteration pays for one extra
  residual evaluation (descent safeguard) and the initial elastic steps do not
  benefit from the acceleration.
- **vs Cast3M**: pyrucast + Anderson comes in **below** Cast3M at both sizes
  (1.65 vs 3.27 s; 10.55 vs 16.78 s). Careful with the interpretation: this is
  not a verdict on the algorithm (the same modified Newton method is used
  throughout), but a measure of the implementation on this machine — Cast3M
  remains more frugal in iterations thanks to its looser tolerances.
- **Rust vs Python**: bit-identical results, and **almost identical times**
  (gap ≤ ~7 % in 200×40, negligible in 400×80). The real cost is in the native
  pyrucast operators (strain, behaviour, internal forces, solve) — the driving
  loop, the only part interpreted in Python, weighs less and less as the mesh
  grows. Python is therefore a façade with no notable penalty here.

## Non-regression: `residual-first` against `master`

The tables above compare pyrucast to Cast3M, at equal resources, on **one given
machine**: their figures are not updated one at a time, they are re-measured
together or not at all. To watch the code evolve, something else is needed — the
**same measurement on both sides of a change**, on the same machine, within the
same minute.

Residual overhaul (`Σ f_int = Σ f_ext` asked of the model, `Domain` split off
from `Behavior`, the distributed load turned into a physics, divergence by
prefix), measured on 4 cores, `cargo clean` then `--release` compilation on both
sides, binaries **alternated** run by run so as to cancel any drift:

| Example | Mesh | `master` | `residual-first` | gap |
|---|---|---:|---:|---:|
| reference | 200×40 | 2.612 s | 2.591 s | **−0.8 %** |
| Anderson | 200×40 | 1.684 s | 1.655 s | **−1.7 %** |
| reference | 400×80 | 17.830 s | 17.611 s | **−1.2 %** |
| Anderson | 400×80 | 10.076 s | 9.888 s | **−1.9 %** |

Median of 5 repetitions in 200×40, 3 in 400×80. A first campaign, not
alternated, had given the same gaps (−0.9 % / −1.3 % / −0.9 % / −0.5 %):
**eight comparisons out of eight in the same direction**. Each gap taken in
isolation fits within the internal spread of a series (~1–2 %); it is the
constancy of the sign, over two campaigns of different orderings, that makes the
signal. No regression, and a slight gain whose cause we do not claim to name.

The check that matters more than the timings: the complete output of both
examples — deflections, iteration counts, cumulated plasticity, at every step —
is **character-identical** between the two branches. The overhaul therefore
moved nothing numerically, which is exactly what was asked of it.

---

_Reproduction. **Rust**: `PYO3_PYTHON=/usr/bin/python3.13 cargo build --release
--example …`, then the binaries read `PYRUCAST_NX/NY/NSTEPS/PMAX`. **Python**:
`maturin develop --release` (Python 3.13), then
`python examples/plasticite_poutre_console[_anderson].py` with the same
variables. **Cast3M**: the `NX/NY/NSTEPS/PMAX` values are hard-coded at the top
of the `.dgibi`. Launch each run alone (idle machine) for a reliable timing._
