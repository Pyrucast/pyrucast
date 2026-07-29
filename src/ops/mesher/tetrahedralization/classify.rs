//! Telling the material from the void.
//!
//! Once every envelope facet is present in the triangulation, the facets act
//! as **walls**: two tetrahedra separated by one are on opposite sides of
//! the surface, and any other pair of neighbours is on the same side. So the
//! inside is found by flooding from cells known to be inside and never
//! crossing a wall — and "known to be inside" comes for free from the
//! orientation the caller supplied, since the material is on the side the
//! facet's normal points away from.
//!
//! Internal cavities need no special handling. A cavity is a closed surface
//! whose normals point into the hole, so the cells filling the hole are
//! seeded as *outside* by exactly the same rule.
//!
//! The flood is run from both sides, and the two results must partition the
//! mesh: every cell inside or outside, none both, none neither. A cell that
//! escapes that is a leak — a facet that recovery failed to install
//! properly — and is reported rather than quietly meshed over.

use std::collections::HashSet;

use crate::error::{PyrucastError, Result};
use crate::interrupt::Cancel;

use super::delaunay::TetMesh;
use super::envelope::Envelope;

/// Which tetrahedra of `mesh` lie in the material bounded by `envelope`.
///
/// Indexed by slot; dead slots are `false`.
pub fn interior(mesh: &TetMesh, envelope: &Envelope, cancel: &dyn Cancel) -> Result<Vec<bool>> {
    let walls: HashSet<[u32; 3]> = envelope
        .facets()
        .iter()
        .map(|f| {
            let mut k = *f;
            k.sort_unstable();
            k
        })
        .collect();

    let mut inside = vec![false; mesh.slot_count()];
    let mut outside = vec![false; mesh.slot_count()];
    let mut seeds_in: Vec<u32> = Vec::new();
    let mut seeds_out: Vec<u32> = Vec::new();

    for f in envelope.facets() {
        cancel.check()?;
        let owners = mesh.face_owners(f).ok_or_else(|| {
            PyrucastError::Message(format!(
                "mesh_volume: envelope facet ({}, {}, {}) is missing from the mesh after \
                 recovery (internal error)",
                f[0], f[1], f[2]
            ))
        })?;
        for (t, i) in owners {
            // The cell's own apex tells which side it is on: the facet's
            // normal points out of the material, so an apex below it is in.
            let apex = mesh.tet(t as usize).expect("live owner")[i];
            let p = mesh.points();
            let side = super::predicates::orient3d(
                &p[f[0] as usize],
                &p[f[1] as usize],
                &p[f[2] as usize],
                &p[apex as usize],
            );
            if side < 0.0 {
                seeds_in.push(t);
            } else if side > 0.0 {
                seeds_out.push(t);
            }
        }
    }
    if seeds_in.is_empty() {
        return Err(PyrucastError::Message(
            "mesh_volume: no tetrahedron lies inside the envelope".into(),
        ));
    }

    flood(mesh, &walls, &seeds_in, &mut inside, cancel)?;
    flood(mesh, &walls, &seeds_out, &mut outside, cancel)?;

    // Cells beyond the envelope but inside the convex hull are reached from
    // the hull's own outer faces too, which have no neighbour at all.
    for (t, _) in mesh.iter() {
        if inside[t] == outside[t] {
            return Err(PyrucastError::Message(format!(
                "mesh_volume: tetrahedron {t} is {} — the recovered envelope does not close \
                 the mesh (internal error)",
                if inside[t] {
                    "reachable from both sides of the envelope"
                } else {
                    "cut off from both sides of the envelope"
                }
            )));
        }
    }
    Ok(inside)
}

/// Spread `reached` from `seeds` over face adjacency, stopping at walls.
fn flood(
    mesh: &TetMesh,
    walls: &HashSet<[u32; 3]>,
    seeds: &[u32],
    reached: &mut [bool],
    cancel: &dyn Cancel,
) -> Result<()> {
    let mut stack: Vec<u32> = Vec::with_capacity(seeds.len());
    for &s in seeds {
        if !reached[s as usize] {
            reached[s as usize] = true;
            stack.push(s);
        }
    }
    while let Some(t) = stack.pop() {
        cancel.check()?;
        for i in 0..4 {
            let mut key = mesh.face(t as usize, i);
            key.sort_unstable();
            if walls.contains(&key) {
                continue;
            }
            if let Some(n) = mesh.neighbour(t as usize, i) {
                if !reached[n] {
                    reached[n] = true;
                    stack.push(n as u32);
                }
            }
        }
    }
    Ok(())
}
