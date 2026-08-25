//! Parallel **scaling** harness — measures the speedup of the hot FE paths
//! (per-element assembly and per-Gauss-point behaviour integration) as the
//! thread count grows.
//!
//! Self-contained and portable: it drives the thread count **itself** (one
//! rayon pool per measurement) so a single run reports the whole curve. No need
//! to juggle `RAYON_NUM_THREADS`.
//!
//! ```text
//! cargo run --release --bin scaling [n] [reps]
//!   n     grid size → n×n QUA4 cells          (default 60)
//!   reps  timed repetitions per thread count  (default 20)
//! ```
//!
//! Output: for each operation, a table `threads | time | speedup | efficiency`,
//! with speedup/efficiency relative to the 1-thread run. Build in **release**
//! (`--release`) — a debug build's numbers are meaningless.

use pyrucast::ops::model;
use std::hint::black_box;
use std::time::{Duration, Instant};

use pyrucast::atoms::{ElementType, Node, NodeId};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::elasticity::ElasticityModel;
use pyrucast::ops::element_field::material_field;
use pyrucast::ops::{element_field, matrix};

/// Plane-stress elasticity on an `n × n` QUA4 grid: model, material field, and a
/// strain field ready for `element_field::behavior::integrate`.
fn build(n: usize) -> (Model, ElementField, ElementField) {
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

    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::QUA4));
    for j in 0..n {
        for i in 0..n {
            mesh.add_cell(&[at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)])
                .unwrap();
        }
    }
    let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    let model = model::elasticity(&fes, ElasticityModel::PlaneStress).unwrap();
    let materials = material_field(&model, &[("E", 210e9), ("nu", 0.3)]).unwrap();

    let support = Handle::new(SubMesh::poi1_from_node_ids(coords.clone(), &ids).unwrap());
    let mut u = SubNodeField::from_poi1(&support, vec!["u_x".into(), "u_y".into()]).unwrap();
    for j in 0..=n {
        for i in 0..=n {
            let nid = at(i, j);
            u.set_value(nid, "u_x", 0.01 * i as f64).unwrap();
            u.set_value(nid, "u_y", -0.005 * j as f64).unwrap();
        }
    }
    let strain = element_field::deformation(&NodeField::from_sub(u), &fes).unwrap();
    (model, materials, strain)
}

/// Median wall time of `reps` runs of `f` executed on a `threads`-wide rayon
/// pool (one warm-up run first).
fn measure(threads: usize, reps: u32, f: &(dyn Fn() + Sync)) -> Duration {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(f); // warm-up
    let start = Instant::now();
    pool.install(|| {
        for _ in 0..reps {
            f();
        }
    });
    start.elapsed() / reps
}

/// Thread counts to sweep: powers of two up to `max`, plus `max` itself.
fn thread_counts(max: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut t = 1;
    while t < max {
        out.push(t);
        t *= 2;
    }
    out.push(max);
    out.dedup();
    out
}

fn run_op(name: &str, threads: &[usize], reps: u32, f: &(dyn Fn() + Sync)) {
    println!("── {name} ──");
    println!("threads        time     speedup   efficiency");
    let mut base: Option<f64> = None;
    for &t in threads {
        let secs = measure(t, reps, f).as_secs_f64();
        let b = *base.get_or_insert(secs);
        let speedup = b / secs;
        let eff = 100.0 * speedup / t as f64;
        println!(
            "{t:>5}   {:>9.3} ms   {:>5.2}x      {:>5.1}%",
            secs * 1e3,
            speedup,
            eff
        );
    }
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let reps: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let max = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(8);
    let threads = thread_counts(max);

    let (model, materials, strain) = build(n);
    let cells = n * n;
    let dofs = 2 * (n + 1) * (n + 1);
    println!(
        "pyrucast scaling — {n}×{n} QUA4 ({cells} cells, {dofs} dofs), \
         reps={reps}, cores={max}\n"
    );

    let stiffness = {
        let model = &model;
        let materials = &materials;
        move || {
            black_box(matrix::stiffness(model, materials).unwrap());
        }
    };
    run_op("matrix::stiffness", &threads, reps, &stiffness);

    let integrate = {
        let model = &model;
        let materials = &materials;
        let strain = &strain;
        move || {
            black_box(
                element_field::behavior::integrate(model, strain, None, materials, None).unwrap(),
            );
        }
    };
    run_op(
        "element_field::behavior::integrate",
        &threads,
        reps,
        &integrate,
    );

    println!(
        "Note: speedup is vs the 1-thread run. Both ops are colour-parallel. \
         matrix::stiffness memoises its CSR sparsity on the model (built once, \
         reused here across reps), so the timed path is the parallel scatter; a \
         first-ever assembly also pays the one-off symbolic build."
    );
}
