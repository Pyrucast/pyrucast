//! Parallelism benchmarks for the hot FE paths.
//!
//! Measures the three heaviest operations on a sizeable mesh:
//!   - `assemble::stiffness` (per-element assembly),
//!   - `behavior::integrate` (per-Gauss-point constitutive law),
//!   - the linear solver, contrasting a **fresh factorization every solve**
//!     against the **transparently cached** factorization (factor once, solve
//!     many).
//!
//! The first two scale with `RAYON_NUM_THREADS`: run the bench twice, e.g.
//! `RAYON_NUM_THREADS=1 cargo bench` then `RAYON_NUM_THREADS=8 cargo bench`,
//! and compare. The solver's faer factorization is multithreaded too; the
//! cached path additionally shows the algorithmic win of reuse.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use pyrucast::aggregate::Aggregate;
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::matrix::{DofOrdering, Matrix, SubMatrix};
use pyrucast::containers::mesh::{Coords, ElementType, Mesh, Node, NodeId, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::models::elasticity::ElasticityModel;
use pyrucast::ops::build::material_field;
use pyrucast::ops::solver::lu::{solve, solve_with_options, SolveMethod, SolveOptions};
use pyrucast::ops::{assemble, behavior, field};
use pyrucast::store::insert;

/// Plane-stress elasticity on an `n × n` QUA4 grid: returns the model, its
/// material field, and a strain field ready to feed `behavior::integrate`.
fn elasticity_grid(n: usize) -> (Model, ElementField, ElementField) {
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

    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    for j in 0..n {
        for i in 0..n {
            mesh.add_cell(&[at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)])
                .unwrap();
        }
    }
    let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    let model = Model::elasticity(&fes, ElasticityModel::PlaneStress).unwrap();
    let materials = material_field(&model, &[("E", 210e9), ("nu", 0.3)]).unwrap();

    // A smooth displacement field u = (0.01·x, −0.005·y) → constant strain.
    let support = insert(SubMesh::poi1_from_node_ids(coords.clone(), &ids).unwrap());
    let mut u = SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into()]).unwrap();
    for j in 0..=n {
        for i in 0..=n {
            let nid = at(i, j);
            u.set_value(nid, "u_x", 0.01 * i as f64).unwrap();
            u.set_value(nid, "u_y", -0.005 * j as f64).unwrap();
        }
    }
    let strain = field::deformation(&NodeField::from_sub(u), &fes).unwrap();
    (model, materials, strain)
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
    let (model, materials, strain) = elasticity_grid(40); // 1600 QUA4 cells
    c.bench_function("stiffness elasticity 40x40 QUA4", |b| {
        b.iter(|| black_box(assemble::stiffness(&model, &materials).unwrap()))
    });
    c.bench_function("integrate elasticity 40x40 QUA4", |b| {
        b.iter(|| black_box(behavior::integrate(&model, &strain, &materials).unwrap()))
    });
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

criterion_group!(benches, bench_assembly, bench_solver);
criterion_main!(benches);
