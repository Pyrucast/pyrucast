//! Mesh-plumbing benchmarks: welding and copying big meshes.
//!
//! These two are what a build script spends its minutes in, long before any
//! finite-element assembly runs. Both are dominated by *bookkeeping* — mapping
//! node ids, moving refcounts, taking the `Coords` lock — rather than by
//! floating-point work, so they are the cases where a per-cell lock or a
//! per-node hash lookup shows up in full.
//!
//! The mesh reproduces the way a solid is usually built by hand: one row of
//! HEX8 meshed once, then copied layer after layer, each copy carrying its own
//! nodes. Consecutive layers share an interface geometrically but not
//! numerically — which is exactly what [`merge_nodes`](pyrucast::ops::mesh)
//! is there to fix, and it makes `(NX+1)·(NY+1)` duplicates per interface.
//!
//! Sizes are chosen so each case runs for about half a second. Anything shorter
//! is drowned by run-to-run noise: on this machine two runs of *identical* code
//! differ by ±18 % at a few milliseconds and by ±20 % at a few tens.

use criterion::{criterion_group, criterion_main, Criterion};
// `black_box` vient de `std` depuis criterion 0.8, qui a déprécié le sien.
use std::hint::black_box;

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, NodeId};
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;
use pyrucast::ops::mesh;

/// Cells per side of one layer, and number of stacked layers: `NX·NY·LAYERS`
/// hexahedra, `(LAYERS−1)·(NX+1)·(NY+1)` nodes to weld away.
const NX: usize = 180;
const NY: usize = 180;
const LAYERS: usize = 32;

/// One layer of `NX × NY` HEX8 cells at height `k`, with its **own** nodes.
fn layer(coords: &Handle<Coords>, k: usize) -> SubMesh {
    let (sx, sy) = (NX + 1, NY + 1);
    let h = 1.0 / LAYERS as f64;
    let mut flat: Vec<f64> = Vec::with_capacity(sx * sy * 2 * 3);
    for level in 0..2 {
        for j in 0..sy {
            for i in 0..sx {
                flat.extend_from_slice(&[
                    i as f64 / NX as f64,
                    j as f64 / NY as f64,
                    (k + level) as f64 * h,
                ]);
            }
        }
    }
    let ids = coords.write().add_nodes(&flat).unwrap();
    let at =
        |i: usize, j: usize, level: usize| NodeId(ids.start + ((level * sy + j) * sx + i) as u32);

    let mut conn: Vec<NodeId> = Vec::with_capacity(NX * NY * 8);
    for j in 0..NY {
        for i in 0..NX {
            conn.extend_from_slice(&[
                at(i, j, 0),
                at(i + 1, j, 0),
                at(i + 1, j + 1, 0),
                at(i, j + 1, 0),
                at(i, j, 1),
                at(i + 1, j, 1),
                at(i + 1, j + 1, 1),
                at(i, j + 1, 1),
            ]);
        }
    }
    let sm = SubMesh::from_connectivity(coords.clone(), ElementType::HEX8, conn).unwrap();
    // The nodes were handed over with one unit each; the connectivity owns
    // them now.
    let owned: Vec<NodeId> = ids.map(NodeId).collect();
    coords.write().decref_all(&owned).unwrap();
    sm
}

/// The stack: one zone per layer, over a single `Coords`.
fn stacked_layers() -> Mesh {
    let coords = Handle::new(Coords::new(3).unwrap());
    let mut stack = Mesh::empty();
    for k in 0..LAYERS {
        stack.add_sub(Handle::new(layer(&coords, k))).unwrap();
    }
    stack
}

fn bench_merge_nodes(c: &mut Criterion) {
    let stack = stacked_layers();
    let cells = stack.cell_count().unwrap();
    c.bench_function(&format!("merge_nodes (copy) on {cells} HEX8"), |bch| {
        bch.iter(|| black_box(mesh::merge_nodes(&stack, 1e-9, false).unwrap()))
    });
}

fn bench_translate(c: &mut Criterion) {
    let stack = stacked_layers();
    let cells = stack.cell_count().unwrap();
    c.bench_function(&format!("translate {cells} HEX8"), |bch| {
        bch.iter(|| black_box(mesh::translate(&stack, &[0.0, 0.0, 1.0]).unwrap()))
    });
}

criterion_group!(benches, bench_merge_nodes, bench_translate);
criterion_main!(benches);
