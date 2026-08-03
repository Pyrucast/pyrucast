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

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node, NodeId};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix};
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::models::elasticity::ElasticityModel;
use pyrucast::ops::element_field;
use pyrucast::ops::element_field::material_field;
use pyrucast::ops::solver::lu::{solve, solve_with_options, SolveMethod, SolveOptions};
use pyrucast::store::insert;

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
fn elasticity_grid_with(n: usize, model_kind: ElasticityModel) -> Grid {
    let coords = insert(if model_kind.is_axisymmetric() {
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
    let model = Model::elasticity(&fespace, model_kind).unwrap();
    let materials = material_field(&model, &[("E", 210e9), ("nu", 0.3)]).unwrap();

    // A smooth displacement field u = (0.01·x, −0.005·y) → constant strain.
    let support = insert(SubMesh::poi1_from_node_ids(coords.clone(), &ids).unwrap());
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
    elasticity_grid_with(n, ElasticityModel::PlaneStress)
}

/// A strictly diagonally-dominant (hence non-singular) 5-point-Laplacian-like
/// system on an `n × n` grid, finalized, plus a unit right-hand side. Used to
/// benchmark the solver without needing boundary conditions.
fn spd_system(n: usize) -> (Matrix, NodeField) {
    let coords = insert(Coords::new(2).unwrap());
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
        insert(sm)
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
    m.add_sub(insert(block)).unwrap();
    m.finalize().unwrap();

    let mut rhs = SubNodeField::from_poi1(&sm, vec!["q".into()]).unwrap();
    for &id in &ids {
        rhs.set_value(id, "q", 1.0).unwrap();
    }
    (m, NodeField::from_sub(rhs))
}

fn bench_assembly(c: &mut Criterion) {
    let g = elasticity_grid(40); // 1600 QUA4 cells
    c.bench_function("stiffness elasticity 40x40 QUA4", |b| {
        b.iter(|| black_box(pyrucast::ops::matrix::stiffness(&g.model, &g.materials).unwrap()))
    });
    c.bench_function("integrate elasticity 40x40 QUA4", |b| {
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
    });
}

/// Plane vs axisymmetric on the same geometry, over the four operations the
/// revolved formulation touches: the stiffness (hoop row of `B`), the
/// deformation (the extra `ε_θθ = u_r/r` component), the constitutive law (four
/// Voigt components instead of three) and the internal forces (the hoop `Bᵀ`).
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
        ("plane", ElasticityModel::PlaneStrain),
        ("axisymmetric", ElasticityModel::Axisymmetric),
    ] {
        let g = elasticity_grid_with(40, kind);
        let state = pyrucast::ops::element_field::behavior::integrate(
            &g.model,
            &g.strain,
            None,
            &g.materials,
            None,
        )
        .unwrap();

        c.bench_function(&format!("stiffness {tag} 40x40 QUA4"), |b| {
            b.iter(|| black_box(pyrucast::ops::matrix::stiffness(&g.model, &g.materials).unwrap()))
        });
        c.bench_function(&format!("deformation {tag} 40x40 QUA4"), |b| {
            b.iter(|| black_box(element_field::deformation(&g.displacement, &g.fespace).unwrap()))
        });
        c.bench_function(&format!("integrate {tag} 40x40 QUA4"), |b| {
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
        });
        c.bench_function(&format!("internal_forces {tag} 40x40 QUA4"), |b| {
            b.iter(|| {
                black_box(pyrucast::ops::node_field::internal_forces(&g.model, &state).unwrap())
            })
        });
    }
}

fn bench_solver(c: &mut Criterion) {
    let n = 30; // 961 unknowns
    let no_cache = SolveOptions {
        method: SolveMethod::Lu,
        cache: false,
    };

    // Fresh factorization on every solve.
    c.bench_function("solve 30x30 fresh factorization", |b| {
        let (m, rhs) = spd_system(n);
        b.iter(|| black_box(solve_with_options(&m, &rhs, &no_cache).unwrap()))
    });

    // Cached factorization: warm once, then every iteration reuses the factors.
    c.bench_function("solve 30x30 cached factorization", |b| {
        let (m, rhs) = spd_system(n);
        let _ = solve(&m, &rhs).unwrap(); // warm the cache
        b.iter(|| black_box(solve(&m, &rhs).unwrap()))
    });
}

criterion_group!(benches, bench_assembly, bench_axisymmetric, bench_solver);
criterion_main!(benches);
