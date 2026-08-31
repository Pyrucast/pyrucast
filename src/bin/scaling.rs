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
//!
//! cargo run --release --bin scaling cube [n] [reps]
//!   n     grid size → n×n×n HEX8 cells, 3-D elasticity (default 20)
//!   reps  timed repetitions of the hot assembly            (default 5)
//! ```
//!
//! Output: for each operation, a table `threads | time | speedup | efficiency`,
//! with speedup/efficiency relative to the 1-thread run. Build in **release**
//! (`--release`) — a debug build's numbers are meaningless.
//!
//! The `cube` mode answers a different question: not « does it scale across
//! cores » but « where does one assembly spend its time and its bytes ». It
//! walks the 3-D path phase by phase — material field, union of zones, cold
//! assembly (symbolic + numeric), hot assembly (sparsity cached) — and prints,
//! beside each timing, the memory each intermediate of the assembler costs at
//! that size. It is the before/after reference for the assembly work.

use pyrucast::ops::model;
use std::hint::black_box;
use std::time::{Duration, Instant};

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Node, NodeId};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::containers::node_field::{NodeField, SubNodeField};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::tensor::Kinematics;
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
    let model = model::elasticity(&fes, Kinematics::PlaneStress).unwrap();
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

// ─── Cube mode: where one 3-D assembly spends its time and its bytes ────────

/// `b` bytes in the largest unit that keeps it readable.
fn bytes(b: usize) -> String {
    const UNITS: [&str; 5] = ["B", "kB", "MB", "GB", "TB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u + 1 < UNITS.len() {
        v /= 1024.0;
        u += 1;
    }
    format!("{v:>7.1} {}", UNITS[u])
}

/// Full 3-D elasticity on an `n × n × n` HEX8 cube: model + uniform material.
///
/// One zone, one computed block — the shape a solid mesh actually assembles in,
/// and the one whose intermediates blow up with the cell count.
fn build_cube(n: usize) -> (Model, ElementField, FiniteElementSpace) {
    let coords = Handle::new(Coords::new(3).unwrap());
    let side = n + 1;
    let mut ids: Vec<NodeId> = Vec::with_capacity(side * side * side);
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                ids.push(
                    Node::create_in(coords.clone(), &[i as f64, j as f64, k as f64])
                        .unwrap()
                        .id(),
                );
            }
        }
    }
    let at = |i: usize, j: usize, k: usize| ids[(k * side + j) * side + i];

    // HEX8: bottom face CCW (0..3), then the matching top face (4..7).
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::HEX8));
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                mesh.add_cell(&[
                    at(i, j, k),
                    at(i + 1, j, k),
                    at(i + 1, j + 1, k),
                    at(i, j + 1, k),
                    at(i, j, k + 1),
                    at(i + 1, j, k + 1),
                    at(i + 1, j + 1, k + 1),
                    at(i, j + 1, k + 1),
                ])
                .unwrap();
            }
        }
    }
    let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
    let model = model::elasticity(&fes, Kinematics::Full3D).unwrap();
    let materials = material_field(&model, &[("E", 210e9), ("nu", 0.3)]).unwrap();
    (model, materials, fes)
}

/// Run `f` once, print its wall time under `name`, and hand back its result.
fn phase<T>(name: &str, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = f();
    println!(
        "{name:<34} {:>10.3} ms",
        start.elapsed().as_secs_f64() * 1e3
    );
    out
}

/// Chain `zones` single-cell material fields with `|`, the way a multi-material
/// model is composed. Quadratic in the zone count today, which is the point of
/// measuring it apart from everything else.
fn union_chain(zones: usize) -> usize {
    let coords = Handle::new(Coords::new(3).unwrap());
    let mut fields: Vec<ElementField> = Vec::with_capacity(zones);
    for z in 0..zones {
        let o = z as f64;
        let n: Vec<NodeId> = [
            [o, 0.0, 0.0],
            [o + 1.0, 0.0, 0.0],
            [o, 1.0, 0.0],
            [o, 0.0, 1.0],
        ]
        .iter()
        .map(|p| Node::create_in(coords.clone(), p).unwrap().id())
        .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::TET4);
        sm.add_cell(&n).unwrap();
        let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();
        let m = model::elasticity(&fes, Kinematics::Full3D).unwrap();
        fields.push(material_field(&m, &[("E", 210e9), ("nu", 0.3)]).unwrap());
    }
    let mut acc = ElementField::empty();
    for f in &fields {
        acc = acc.union(f).unwrap();
    }
    acc.len()
}

/// Walk one 3-D assembly phase by phase, then print what each intermediate of
/// the assembler costs in memory at this size.
fn run_cube(n: usize, reps: u32) {
    let cells = n * n * n;
    let nodes = (n + 1) * (n + 1) * (n + 1);
    let dofs = 3 * nodes;
    let ke_len = 24 * 24; // HEX8 × 3 ddl
    let max = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(8);
    println!(
        "pyrucast scaling — cube {n}×{n}×{n} HEX8, élasticité 3-D \
         ({cells} mailles, {dofs} ddl), reps={reps}, cores={max}\n"
    );

    println!("── phases ──");
    let (model, materials, fespace) = phase("build_cube (maillage + EF)", || build_cube(n));
    let mat_again = phase("element_field::material_field", || {
        material_field(&model, &[("E", 210e9), ("nu", 0.3)]).unwrap()
    });
    black_box(&mat_again);
    const ZONES: usize = 256;
    let fused = phase("union en chaîne (256 zones)", || union_chain(ZONES));
    debug_assert_eq!(fused, ZONES);

    // Two component-disjoint material zones on one support, then fused: the
    // path a multi-physics material takes, and the one that walks every Gauss
    // point of the mesh.
    let deux = ElementField::new(&fespace, vec!["E".into(), "nu".into()])
        .unwrap()
        .union(&ElementField::new(&fespace, vec!["rho".into()]).unwrap())
        .unwrap();
    let une = phase("element_field::consolidate", || {
        element_field::consolidate(&deux).unwrap()
    });
    debug_assert_eq!(une.len(), 1);

    // The first assembly pays the symbolic phase (DOF numbering + sparsity);
    // every later one reuses the pattern memoised on the model.
    let k = phase("matrix::stiffness (à froid)", || {
        matrix::stiffness(&model, &materials).unwrap()
    });
    let nnz = k.to_csr().unwrap().nnz();
    drop(k);

    let start = Instant::now();
    for _ in 0..reps {
        black_box(matrix::stiffness(&model, &materials).unwrap());
    }
    let hot = start.elapsed().as_secs_f64() / reps as f64;
    println!(
        "{:<34} {:>10.3} ms",
        "matrix::stiffness (à chaud)",
        hot * 1e3
    );

    println!("\n── mémoire des intermédiaires (nnz = {nnz}) ──");
    let row = |name: &str, b: usize| println!("{name:<34} {}", bytes(b));
    // Ce que l'assembleur alloue réellement aujourd'hui.
    row("motif : position des nœuds", 2 * cells * 8 * 4);
    row("motif : colonnes avant dédup", cells * ke_len * 4);
    row("block_slots (bases par nœud)", cells * (8 * 3 * 8) * 4);
    row("CSR : valeurs", nnz * 8);
    row("CSR : col_indices", nnz * std::mem::size_of::<usize>());
    row("tampon atomique du scatter", nnz * 8);
    // Ce qu'il allouait avant, pour l'échelle.
    println!(
        "\n   (avant : paires (r,c) {} ; block_slots {} ; ke toutes matérialisées {})",
        bytes(cells * ke_len * std::mem::size_of::<(usize, usize)>()),
        bytes(cells * ke_len * std::mem::size_of::<usize>()),
        bytes(cells * ke_len * 8)
    );

    println!(
        "\nNote : « à froid » inclut la numérotation des ddl et le motif de \
         sparsité ; « à chaud » réutilise le motif mémoïsé sur le modèle et ne \
         mesure donc que la phase numérique."
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("cube") {
        let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
        let reps: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
        run_cube(n, reps);
        return;
    }
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
