//! Geometric-query benchmarks: inverting the element mapping.
//!
//! [`locate_points`](pyrucast::ops::geom::locate_points) and
//! [`project_points`](pyrucast::ops::geom::project_points) are the two
//! operations that evaluate shape functions **inside a Newton loop** — up to 40
//! iterations per candidate cell, per point — rather than reading them from the
//! tables an FE space precomputes. They are what the `embedded` constraint and
//! node-to-surface contact are built on, and the only paths where the cost of
//! `ElementKind::shape_into` is visible at all.
//!
//! Everything else in the FE pipeline (assembly, integration, internal forces)
//! works off `SubFiniteElementSpace`'s precomputed `n_at_g` / `dn_at_g`, so it
//! never calls a shape function per cell; see `benches/parallel.rs`.
//!
//! Sizes are chosen so each case runs for about half a second. Anything shorter
//! is drowned by run-to-run noise: on this machine two runs of *identical* code
//! differ by ±18 % at a few milliseconds and by ±20 % at a few tens.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use pyrucast::atoms::{ElementType, Node, NodeId};
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;

/// Host block side, in cells: `HOST_N³` hexahedra.
const HOST_N: usize = 12;
/// Points located inside the block. Sized for a ~0.5 s run: locating one point
/// costs well under a microsecond, so it takes a lot of them.
const N_POINTS: usize = 900_000;
/// Surface side, in cells, for the projection case.
const SURFACE_N: usize = 40;
/// Points projected onto the surface. Far fewer than for `locate`:
/// `project_points` has no spatial index and tries every facet, so it costs
/// some 300 µs per point on this mesh.
const N_PROJECTED: usize = 2_000;

/// An `n × n × n` block of HEX8 cells spanning the unit cube.
fn hex_block(n: usize) -> Mesh {
    let coords = Handle::new(Coords::new(3).unwrap());
    let side = n + 1;
    let mut ids: Vec<NodeId> = Vec::with_capacity(side * side * side);
    for k in 0..side {
        for j in 0..side {
            for i in 0..side {
                let p = [
                    i as f64 / n as f64,
                    j as f64 / n as f64,
                    k as f64 / n as f64,
                ];
                ids.push(Node::create_in(coords.clone(), &p).unwrap().id());
            }
        }
    }
    let at = |i: usize, j: usize, k: usize| ids[(k * side + j) * side + i];
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::HEX8));
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
    mesh
}

/// An `n × n` QUA4 surface in 3-D, gently curved so the projection is a real
/// Newton solve rather than an exactly-linear one.
fn curved_surface(n: usize) -> Mesh {
    let coords = Handle::new(Coords::new(3).unwrap());
    let side = n + 1;
    let mut ids: Vec<NodeId> = Vec::with_capacity(side * side);
    for j in 0..side {
        for i in 0..side {
            let (x, y) = (i as f64 / n as f64, j as f64 / n as f64);
            let z = 0.2 * (x - 0.5) * (y - 0.5);
            ids.push(Node::create_in(coords.clone(), &[x, y, z]).unwrap().id());
        }
    }
    let at = |i: usize, j: usize| ids[j * side + i];
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::QUA4));
    for j in 0..n {
        for i in 0..n {
            mesh.add_cell(&[at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)])
                .unwrap();
        }
    }
    mesh
}

/// A deterministic scatter of points, so two runs measure the same work.
fn scatter(count: usize, z: impl Fn(f64, f64) -> f64) -> Vec<Vec<f64>> {
    // A cheap low-discrepancy-ish sequence: irrational strides mod 1.
    let mut out = Vec::with_capacity(count);
    let (mut a, mut b) = (0.5f64, 0.5f64);
    for _ in 0..count {
        a = (a + std::f64::consts::FRAC_1_SQRT_2).fract();
        b = (b + 0.618_033_988_749_895).fract();
        out.push(vec![a, b, z(a, b)]);
    }
    out
}

fn bench_locate(c: &mut Criterion) {
    let host = hex_block(HOST_N);
    let pts = scatter(N_POINTS, |a, b| (a * b).fract());
    c.bench_function(
        &format!("locate {N_POINTS} points in {HOST_N}³ HEX8"),
        |bch| {
            bch.iter(|| black_box(pyrucast::ops::geom::locate_points(&host, &pts, 1e-9).unwrap()))
        },
    );
}

fn bench_project(c: &mut Criterion) {
    let surface = curved_surface(SURFACE_N);
    // Points floating above the surface, so each one needs a real projection.
    let pts = scatter(N_PROJECTED, |a, b| 0.2 * (a - 0.5) * (b - 0.5) + 0.05);
    c.bench_function(
        &format!("project {N_PROJECTED} points onto {SURFACE_N}² QUA4"),
        |bch| bch.iter(|| black_box(pyrucast::ops::geom::project_points(&surface, &pts).unwrap())),
    );
}

criterion_group!(benches, bench_locate, bench_project);
criterion_main!(benches);
