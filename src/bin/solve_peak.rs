//! **Memory** harness for the direct solve — where a factorization spends its
//! bytes, and how much of that is the matrix rather than the factors.
//!
//! ```text
//! cargo run --release --bin solve_peak -- <mode> [n]
//!   assemble   stop after `stiffness`  — the matrix alone
//!   solve      bit-level fingerprint of a constrained solve — the acceptance
//!              check that a change to the faer handover changed no result
//!   borrowed   a counting-sort transpose, then a borrowed `SparseColMatRef`
//!              handed to `sp_lu` — what a **non-symmetric** matrix gets
//!   direct     no transpose at all: a symmetric matrix is its own CSC, so its
//!              CSR arrays go straight to `sp_lu`
//!   cholesky   the same arrays handed to `sp_cholesky` — what a symmetric
//!              matrix gets when the caller asks for it
//!   triplets   the path it took before: `to_csc`, unfold into `Vec<Triplet>`,
//!              let faer sort them back, own the result — kept as the
//!              before/after reference
//!   n          grid size → n×n×n HEX8 cells, 3-D elasticity (default 20)
//! ```
//!
//! **One mode per process.** `VmHWM` is a high-water mark that only ever rises,
//! so running two paths in one process would report the worse of the two for
//! both.
//!
//! The solve modes mirror `ops::solver::lu` rather than calling it: its entry
//! points are `pub(crate)`, and a binary is a separate crate. They are kept
//! faithful to it by hand — if that module's handover to faer changes, they
//! change with it.
//!
//! A cube carries no boundary condition, so its stiffness is only *semi*
//! definite — the rigid-body modes make it singular. `cholesky` is therefore
//! expected to be **refused** on it, which is itself the point: that refusal is
//! exactly what a caller who picks `method="cholesky"` gets back, naming the
//! pivot where it stopped.
//!
//! **A cube is the worst case for fill.** A massive geometry is where a sparse
//! LU behaves at its worst; a thin structure (sheet, shell, slender part) sits
//! far closer to the 2-D regime, where the factors cost `O(N log N)` instead of
//! `O(N^4/3)`. Read the numbers below as an upper bound on a real mesh of the
//! same size, not as a prediction for one.

use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef, Triplet};
use pyrucast::atoms::{ElementType, Node, NodeId};
use pyrucast::containers::element_field::ElementField;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::matrix::Matrix;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::containers::model::Model;
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::models::tensor::Kinematics;
use pyrucast::ops::element_field::material_field;
use pyrucast::ops::{matrix, model};
use std::time::Instant;

/// `b` bytes in the largest unit that keeps it readable.
fn bytes(b: usize) -> String {
    const UNITS: [&str; 5] = ["B", "kB", "MB", "GB", "TB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u + 1 < UNITS.len() {
        v /= 1024.0;
        u += 1;
    }
    format!("{v:>8.2} {}", UNITS[u])
}

/// One `/proc/self/status` field, in bytes. Zero where the file is absent.
fn vm(key: &str) -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<usize>().ok())
        })
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

/// Reset the peak-RSS watermark to the current RSS, so the next stage's peak is
/// its own rather than the run's. A no-op where the kernel refuses it — the
/// report then reads as a running maximum, which is still sound, just blunter.
fn reset_peak() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

/// Run one stage: its wall time, the RSS it leaves behind, and the peak it
/// reached getting there.
fn stage<T>(name: &str, f: impl FnOnce() -> T) -> T {
    reset_peak();
    let before = vm("VmRSS:");
    let t = Instant::now();
    let out = f();
    let dt = t.elapsed();
    println!(
        "{name:<32} {:>9.2?}   RSS {} (+{})   peak {}",
        dt,
        bytes(vm("VmRSS:")),
        bytes(vm("VmRSS:").saturating_sub(before)),
        bytes(vm("VmHWM:"))
    );
    out
}

/// Full 3-D elasticity on an `n × n × n` HEX8 cube — one zone, one computed
/// block, the shape a solid mesh actually assembles in.
fn build_cube(n: usize) -> (Model, ElementField) {
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
    (model, materials)
}

/// CSR → CSC by counting sort, as `ops::solver::lu::transpose_to_csc` does.
fn transpose_to_csc(
    n: usize,
    offsets: &[usize],
    cols: &[usize],
    vals: &[f64],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let nnz = cols.len();
    let mut col_ptr = vec![0usize; n + 1];
    for &c in cols {
        col_ptr[c + 1] += 1;
    }
    for j in 0..n {
        col_ptr[j + 1] += col_ptr[j];
    }
    let mut cursor = col_ptr[..n].to_vec();
    let mut row_idx = vec![0usize; nnz];
    let mut out = vec![0.0f64; nnz];
    for r in 0..n {
        for k in offsets[r]..offsets[r + 1] {
            let j = cols[k];
            let p = cursor[j];
            row_idx[p] = r;
            out[p] = vals[k];
            cursor[j] = p + 1;
        }
    }
    (col_ptr, row_idx, out)
}

/// A constrained thermal bar of `n` SEG2, solved through the public
/// `solver::lu::solve`, reported as a hash of the **bit patterns** of its
/// solution.
///
/// The acceptance check for any change to the handover to faer: what reaches
/// `sp_lu` must be the same matrix, so what comes back must be the same bits.
/// A tolerance would hide exactly the drift worth catching, hence the hash.
fn solve_fingerprint(n: usize) -> (usize, u64) {
    use pyrucast::aggregate::Aggregate;
    use pyrucast::containers::node_field::NodeField;
    use pyrucast::models::RelationSense;
    use pyrucast::ops::{mesh, solver};

    let coords = Handle::new(Coords::new(1).unwrap());
    let nodes: Vec<Node> = (0..=n)
        .map(|i| Node::create_in(coords.clone(), &[i as f64 / n as f64]).unwrap())
        .collect();
    let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
    for i in 0..n {
        sm.add_cell(&[nodes[i].id(), nodes[i + 1].id()]).unwrap();
    }
    let fes = FiniteElementSpace::lagrange1(&Mesh::from_submesh(sm)).unwrap();

    let imposed = mesh::poi1_from_nodes(std::slice::from_ref(&nodes[0])).unwrap();
    let mult = mesh::barycenter(&imposed).unwrap();
    let mult_node = mult.node(0, 0, 0).unwrap();
    let conduction = model::heat_conduction(&fes).unwrap();
    let m = conduction
        .union(
            &model::dirichlet(&conduction, "T", &imposed, &mult, RelationSense::Equality).unwrap(),
        )
        .unwrap();
    let materials = material_field(&m, &[("k", 1.0)]).unwrap();
    let k = matrix::stiffness(&m, &materials).unwrap();

    let rhs = NodeField::from_submesh(&mult.get(0).unwrap(), vec!["imposed_T".into()]).unwrap();
    rhs.get(0)
        .unwrap()
        .write()
        .set_value(mult_node.id(), "imposed_T", 1.0)
        .unwrap();

    let u = solver::lu::solve(&k, &rhs).unwrap();
    // FNV-1a over each value's raw bits, in node order — deterministic, and
    // sensitive to the last bit.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut counted = 0usize;
    let zone = u.get(0).unwrap();
    let guard = zone.read();
    for node in &nodes {
        if let Ok(v) = guard.value(node.id(), "T") {
            for b in v.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            counted += 1;
        }
    }
    (counted, h)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("assemble");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);

    if mode == "solve" {
        let n = if n == 20 { 400 } else { n };
        let (count, h) = solve_fingerprint(n);
        println!("solve fingerprint: {count} value(s), FNV-1a = {h:#018x}");
        return;
    }

    println!("\n=== {mode} — {n}×{n}×{n} HEX8 cube, 3-D elasticity ===");
    let (model, materials) = stage("mesh + model", || build_cube(n));
    let k: Matrix = stage("assembly (stiffness)", || {
        matrix::stiffness(&model, &materials).unwrap()
    });

    let (offsets, cols, vals) = k.csr_arrays().unwrap();
    let (ndof, nnz) = (k.n_rows().unwrap(), vals.len());
    println!(
        "\n  {ndof} DOF   {nnz} nnz   assembled CSR {}   ({:.1} nnz/row)\n",
        bytes(nnz * 16 + (ndof + 1) * 8),
        nnz as f64 / ndof as f64
    );

    match mode {
        "assemble" => {}
        "borrowed" => {
            let (col_ptr, row_idx, tvals) = stage("transpose (counting sort)", || {
                transpose_to_csc(ndof, offsets, cols, vals)
            });
            let lu = stage("sp_lu (borrowed input)", || {
                let sym = SymbolicSparseColMatRef::<usize>::new_checked(
                    ndof, ndof, &col_ptr, None, &row_idx,
                );
                SparseColMatRef::<usize, f64>::new(sym, &tvals)
                    .sp_lu()
                    .unwrap()
            });
            std::hint::black_box(&lu);
        }
        "triplets" => {
            let csc = stage("to_csc()", || k.to_csc().unwrap());
            let lu = stage("triplets + SparseColMat + sp_lu", || {
                let (co, ri, v) = (csc.col_offsets(), csc.row_indices(), csc.values());
                let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::with_capacity(v.len());
                for col in 0..ndof {
                    for kk in co[col]..co[col + 1] {
                        triplets.push(Triplet::new(ri[kk], col, v[kk]));
                    }
                }
                faer::sparse::SparseColMat::<usize, f64>::try_new_from_triplets(
                    ndof, ndof, &triplets,
                )
                .unwrap()
                .sp_lu()
                .unwrap()
            });
            std::hint::black_box(&lu);
        }
        "direct" => {
            // A symmetric matrix is its own CSC: nothing to turn around.
            let lu = stage("sp_lu (no transpose)", || {
                let sym =
                    SymbolicSparseColMatRef::<usize>::new_checked(ndof, ndof, offsets, None, cols);
                SparseColMatRef::<usize, f64>::new(sym, vals)
                    .sp_lu()
                    .unwrap()
            });
            std::hint::black_box(&lu);
        }
        "cholesky" => {
            let outcome = stage("sp_cholesky (no transpose)", || {
                let sym =
                    SymbolicSparseColMatRef::<usize>::new_checked(ndof, ndof, offsets, None, cols);
                SparseColMatRef::<usize, f64>::new(sym, vals).sp_cholesky(faer::Side::Lower)
            });
            match outcome {
                Ok(llt) => {
                    std::hint::black_box(&llt);
                    println!("  Cholesky accepted this matrix");
                }
                Err(e) => println!("  Cholesky refused: {e:?} — the solver falls back to LU"),
            }
        }
        other => {
            eprintln!(
                "unknown mode '{other}' \
                 (assemble | solve | borrowed | direct | cholesky | triplets)"
            );
            std::process::exit(2);
        }
    }

    println!("\n  -- process peak: {} --\n", bytes(vm("VmHWM:")));
}
