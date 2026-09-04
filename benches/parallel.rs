//! Parallelism benchmarks for the hot FE paths.
//!
//! Measures the three heaviest operations on a sizeable mesh:
//!   - `pyrucast::ops::matrix::stiffness` (per-element assembly),
//!   - `pyrucast::ops::element_field::behavior::integrate` (per-Gauss-point constitutive law),
//!   - the linear solver, contrasting a **fresh factorization every solve**
//!     against the **transparently cached** factorization (factor once, solve
//!     many).
//!
//! The first two scale with `RAYON_NUM_THREADS`: run the bench twice, e.g.
//! `RAYON_NUM_THREADS=1 cargo bench` then `RAYON_NUM_THREADS=8 cargo bench`,
//! and compare. The solver's faer factorization is multithreaded too; the
//! cached path additionally shows the algorithmic win of reuse.
//!
//! A fourth group contrasts the **plane and axisymmetric** formulations on the
//! *same* geometry, over the four operations the revolved case touches. The gap
//! is the cost of the formulation itself — four Voigt components instead of
//! three, plus the `2πr` measure — not an overhead paid by plane models: the
//! Cartesian path takes one predictable branch in
//! `CellGeom::det_j_w` and nothing else.

use criterion::{criterion_group, criterion_main, Criterion};
use pyrucast::ops::model;
// `black_box` vient de `std` depuis criterion 0.8, qui a déprécié le sien.
use std::hint::black_box;

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node, NodeId};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix};
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::tensor::Kinematics;
use pyrucast::ops::element_field;
use pyrucast::ops::element_field::material_field;
use pyrucast::ops::solver::lu::{solve, solve_with_options, SolveMethod, SolveOptions};

/// Grid side for the assembly / integration groups.
///
/// Sized so the heaviest operation of the group runs for about half a second,
/// which is the useful compromise: a benchmark finishing in a few milliseconds
/// cannot be trusted here (two runs of *identical* code differ by ±18 % at that
/// scale, and still by ±20 % at a few tens), while a second-long assembly costs
/// several gigabytes of resident memory. Only `behavior::integrate` stays short
/// — it is some fourteen times cheaper per cell than the assembly; read it with
/// that in mind.
const ASSEMBLY_N: usize = 450; // 202 500 QUA4 cells

/// Grid side for the constitutive-law group.
///
/// `behavior::integrate` is roughly thirteen times cheaper per cell than the
/// assembly, so on the assembly's own grid it ran in some 40 ms — short enough
/// that run-to-run noise reached ±13 %, and the figure said nothing. It gets
/// its own, much larger mesh instead; it allocates no global matrix, so the
/// memory stays reasonable where a stiffness of that size would not.
const INTEGRATE_N: usize = 1900; // 3 610 000 QUA4 cells

/// Grid side for the solver group, sized on the same principle. The direct
/// factorisation grows much faster than the assembly, so it needs its own,
/// smaller mesh.
const SOLVER_N: usize = 250; // 63 001 unknowns

/// Everything an elasticity benchmark needs on one grid.
struct Grid {
    model: Model,
    materials: ElementField,
    fespace: FiniteElementSpace,
    /// Nodal displacement — the input of `element_field::deformation`.
    displacement: NodeField,
    /// `deformation(displacement)` — the input of `pyrucast::ops::element_field::behavior::integrate`.
    strain: ElementField,
}

/// Elasticity on an `n × n` QUA4 grid under the given 2-D `model`.
///
/// The grid starts at `x = 1` so it is a valid meridian plane (`x = r ≥ 0`) and
/// the plane and axisymmetric variants share the **exact same geometry** — a
/// translation leaves the Cartesian Jacobian untouched, so it costs the plane
/// case nothing and makes the two directly comparable.
fn elasticity_grid_with(n: usize, model_kind: Kinematics) -> Grid {
    let coords = Handle::new(if model_kind.is_axisymmetric() {
        Coords::axisymmetric().unwrap()
    } else {
        Coords::new(2).unwrap()
    });
    let mut ids: Vec<NodeId> = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        for i in 0..=n {
            ids.push(
                Node::create_in(coords.clone(), &[1.0 + i as f64, j as f64])
                    .unwrap()
                    .id(),
            );
        }
    }
    let at = |i: usize, j: usize| ids[j * (n + 1) + i];

    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    for j in 0..n {
        for i in 0..n {
            mesh.add_cell(&[at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)])
                .unwrap();
        }
    }
    let fespace = FiniteElementSpace::lagrange1(&mesh).unwrap();
    let model = model::elasticity(&fespace, model_kind).unwrap();
    let materials = material_field(&model, &[("E", 210e9), ("nu", 0.3)]).unwrap();

    // A smooth displacement field u = (0.01·x, −0.005·y) → constant strain.
    let support = Handle::new(SubMesh::poi1_from_node_ids(coords.clone(), &ids).unwrap());
    let mut u = SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into()]).unwrap();
    for j in 0..=n {
        for i in 0..=n {
            let nid = at(i, j);
            u.set_value(nid, "u_x", 0.01 * (1.0 + i as f64)).unwrap();
            u.set_value(nid, "u_y", -0.005 * j as f64).unwrap();
        }
    }
    let displacement = NodeField::from_sub(u);
    let strain = element_field::deformation(&displacement, &fespace).unwrap();
    Grid {
        model,
        materials,
        fespace,
        displacement,
        strain,
    }
}

/// Plane-stress elasticity on an `n × n` QUA4 grid — the default case.
fn elasticity_grid(n: usize) -> Grid {
    elasticity_grid_with(n, Kinematics::PlaneStress)
}

/// A strictly diagonally-dominant (hence non-singular) 5-point-Laplacian-like
/// system on an `n × n` grid, finalized, plus a unit right-hand side. Used to
/// benchmark the solver without needing boundary conditions.
fn spd_system(n: usize) -> (Matrix, NodeField) {
    let coords = Handle::new(Coords::new(2).unwrap());
    let mut ids: Vec<NodeId> = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        for i in 0..=n {
            ids.push(
                Node::create_in(coords.clone(), &[i as f64, j as f64])
                    .unwrap()
                    .id(),
            );
        }
    }
    let at = |i: usize, j: usize| ids[j * (n + 1) + i];

    let sm = {
        let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
        for &id in &ids {
            sm.add_cell(&[id]).unwrap();
        }
        Handle::new(sm)
    };
    let mut block = SubMatrix::new(
        sm.clone(),
        sm.clone(),
        vec!["q".into()],
        vec!["T".into()],
        DofOrdering::NodesThenVars,
        true,
    )
    .unwrap();
    for j in 0..=n {
        for i in 0..=n {
            let c = at(i, j);
            let mut neighbours: Vec<NodeId> = Vec::with_capacity(4);
            if i > 0 {
                neighbours.push(at(i - 1, j));
            }
            if i < n {
                neighbours.push(at(i + 1, j));
            }
            if j > 0 {
                neighbours.push(at(i, j - 1));
            }
            if j < n {
                neighbours.push(at(i, j + 1));
            }
            for &m in &neighbours {
                block.add_entry(c, "q", m, "T", -1.0).unwrap();
            }
            // diag = #neighbours + 1 ⇒ strict diagonal dominance ⇒ non-singular.
            block
                .add_entry(c, "q", c, "T", neighbours.len() as f64 + 1.0)
                .unwrap();
        }
    }
    let mut m = Matrix::empty();
    m.add_sub(Handle::new(block)).unwrap();
    m.finalize().unwrap();

    let mut rhs = SubNodeField::from_poi1(&sm, vec!["q".into()]).unwrap();
    for &id in &ids {
        rhs.set_value(id, "q", 1.0).unwrap();
    }
    (m, NodeField::from_sub(rhs))
}

fn bench_assembly(c: &mut Criterion) {
    let g = elasticity_grid(ASSEMBLY_N);
    c.bench_function(
        &format!("stiffness elasticity {ASSEMBLY_N}x{ASSEMBLY_N} QUA4"),
        |b| b.iter(|| black_box(pyrucast::ops::matrix::stiffness(&g.model, &g.materials).unwrap())),
    );
}

/// The constitutive law on its own grid, big enough for the figure to mean
/// something — see [`INTEGRATE_N`]. Plane and axisymmetric side by side, as in
/// [`bench_axisymmetric`]: the revolved law carries four Voigt components
/// instead of three.
fn bench_integrate(c: &mut Criterion) {
    for (tag, kind) in [
        ("plane", Kinematics::PlaneStrain),
        ("axisymmetric", Kinematics::Axisymmetric),
    ] {
        let g = elasticity_grid_with(INTEGRATE_N, kind);
        c.bench_function(
            &format!("integrate {tag} {INTEGRATE_N}x{INTEGRATE_N} QUA4"),
            |b| {
                b.iter(|| {
                    black_box(
                        pyrucast::ops::element_field::behavior::integrate(
                            &g.model,
                            &g.strain,
                            None,
                            &g.materials,
                            None,
                        )
                        .unwrap(),
                    )
                })
            },
        );
    }
}

/// Plane vs axisymmetric on the same geometry, over the four operations the
/// revolved formulation touches: the stiffness (hoop row of `B`), the
/// deformation (the extra `ε_θθ = u_r/r` component) and the internal forces
/// (the hoop `Bᵀ`). The constitutive law is measured in [`bench_integrate`],
/// which gives it a grid where its cost is resolvable.
///
/// Read the pairs as a ratio, not an absolute: the axisymmetric case does
/// genuinely more arithmetic. A plane model must stay level with the pre-existing
/// `bench_assembly` figures — that is the regression this group guards.
///
/// The ratio itself is size-dependent: at 1600 cells everything is cache-resident
/// and the revolved case measures ~+10 to +35 %, whereas on a memory-bound grid
/// (10⁵–10⁶ cells) it settles around +4 to +20 %. Compare a run against a previous
/// run of the *same* size, not against those figures.
fn bench_axisymmetric(c: &mut Criterion) {
    for (tag, kind) in [
        ("plane", Kinematics::PlaneStrain),
        ("axisymmetric", Kinematics::Axisymmetric),
    ] {
        let g = elasticity_grid_with(ASSEMBLY_N, kind);
        let state = pyrucast::ops::element_field::behavior::integrate(
            &g.model,
            &g.strain,
            None,
            &g.materials,
            None,
        )
        .unwrap();

        c.bench_function(
            &format!("stiffness {tag} {ASSEMBLY_N}x{ASSEMBLY_N} QUA4"),
            |b| {
                b.iter(|| {
                    black_box(pyrucast::ops::matrix::stiffness(&g.model, &g.materials).unwrap())
                })
            },
        );
        c.bench_function(
            &format!("deformation {tag} {ASSEMBLY_N}x{ASSEMBLY_N} QUA4"),
            |b| {
                b.iter(|| {
                    black_box(element_field::deformation(&g.displacement, &g.fespace).unwrap())
                })
            },
        );
        c.bench_function(
            &format!("internal_forces {tag} {ASSEMBLY_N}x{ASSEMBLY_N} QUA4"),
            |b| {
                b.iter(|| {
                    black_box(
                        pyrucast::ops::node_field::internal_forces(
                            &g.model,
                            &state,
                            &g.displacement,
                            &g.materials,
                        )
                        .unwrap(),
                    )
                })
            },
        );
    }
}

fn bench_solver(c: &mut Criterion) {
    let n = SOLVER_N;
    let no_cache = SolveOptions {
        method: SolveMethod::Lu,
        cache: false,
    };

    // Fresh factorization on every solve.
    c.bench_function(
        &format!("solve {SOLVER_N}x{SOLVER_N} fresh factorization"),
        |b| {
            let (m, rhs) = spd_system(n);
            b.iter(|| black_box(solve_with_options(&m, &rhs, &no_cache).unwrap()))
        },
    );

    // Cached factorization: warm once, then every iteration reuses the factors.
    c.bench_function(
        &format!("solve {SOLVER_N}x{SOLVER_N} cached factorization"),
        |b| {
            let (m, rhs) = spd_system(n);
            let _ = solve(&m, &rhs).unwrap(); // warm the cache
            b.iter(|| black_box(solve(&m, &rhs).unwrap()))
        },
    );
}

criterion_group!(
    benches,
    bench_assembly,
    bench_integrate,
    bench_axisymmetric,
    bench_solver
);
criterion_main!(benches);
