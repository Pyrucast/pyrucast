//! Export a mesh or a field to a **legacy VTK** file (`UNSTRUCTURED_GRID`,
//! ASCII) for viewing in ParaView.
//!
//! The legacy `.vtk` format is the simplest ParaView reads natively: a text
//! header, then `POINTS` / `CELLS` / `CELL_TYPES`, then optional
//! `POINT_DATA` (a [`NodeField`]) or `CELL_DATA` (an [`ElementField`]).
//!
//! - **Geometry.** Every submesh of the [`Mesh`] is written; the nodes it
//!   references become VTK points (deduplicated, padded to 3-D with `z = 0`
//!   for a 2-D `Coords`). Cell types map one-to-one and the local node
//!   ordering already matches VTK's, so connectivity is copied verbatim:
//!
//!   | [`ElementType`] | VTK cell | code |
//!   |---|---|---|
//!   | `POI1` | `VERTEX`       | 1  |
//!   | `SEG2` | `LINE`         | 3  |
//!   | `TRI3` | `TRIANGLE`     | 5  |
//!   | `QUA4` | `QUAD`         | 9  |
//!   | `TET4` | `TETRA`        | 10 |
//!   | `PENTA6` | `WEDGE`      | 13 |
//!   | `HEX8` | `HEXAHEDRON`   | 12 |
//!   | `SEG3` | `QUADRATIC_EDGE`     | 21 |
//!   | `TRI6` | `QUADRATIC_TRIANGLE` | 22 |
//!   | `QUA8` | `QUADRATIC_QUAD`     | 23 |
//!   | `TET10` | `QUADRATIC_TETRA`   | 24 |
//!   | `HEX20` | `QUADRATIC_HEXAHEDRON` | 25 |
//!   | `PENTA15` | `QUADRATIC_WEDGE` | 26 |
//!   | `QUA9` | `BIQUADRATIC_QUAD` | 28 |
//!   | `HEX27` | `TRIQUADRATIC_HEXAHEDRON` | 29 |
//!
//! - **Node field** → `POINT_DATA`: one `SCALARS` array per component, the
//!   nodal value at each point (`0` where the field does not define it).
//! - **Element field** → `CELL_DATA`: one `SCALARS` array per component, the
//!   **per-cell mean of that cell's Gauss values** (an intra-element
//!   average — inter-element discontinuities stay visible, one value per
//!   cell). The field must come from a space built on the **same** mesh
//!   (its cells line up one-to-one, submesh by submesh).

use crate::aggregate::Aggregate;
use crate::containers::element_field::ElementField;
use crate::containers::field::{Field, SubField};
use crate::containers::mesh::{ElementType, Mesh, NodeId, SubMesh};
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};
use crate::parallel::*;
use crate::store::{read, Handle};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

/// VTK legacy cell-type code for a pyrucast element type.
fn vtk_cell_type(et: ElementType) -> u8 {
    match et {
        ElementType::POI1 => 1,
        ElementType::SEG2 => 3,
        ElementType::TRI3 => 5,
        ElementType::QUA4 => 9,
        ElementType::TET4 => 10,
        ElementType::PENTA6 => 13,
        ElementType::HEX8 => 12,
        // Quadratic types: pyrucast's node order already matches VTK's, so
        // the connectivity is written verbatim (corners then mid-edges).
        ElementType::SEG3 => 21,    // VTK_QUADRATIC_EDGE
        ElementType::TRI6 => 22,    // VTK_QUADRATIC_TRIANGLE
        ElementType::QUA8 => 23,    // VTK_QUADRATIC_QUAD
        ElementType::TET10 => 24,   // VTK_QUADRATIC_TETRA
        ElementType::HEX20 => 25,   // VTK_QUADRATIC_HEXAHEDRON
        ElementType::PENTA15 => 26, // VTK_QUADRATIC_WEDGE
        ElementType::QUA9 => 28,    // VTK_BIQUADRATIC_QUAD
        ElementType::HEX27 => 29,   // VTK_TRIQUADRATIC_HEXAHEDRON
    }
}

/// VTK array names cannot carry spaces; swap them for underscores.
fn sanitize(name: &str) -> String {
    name.replace(char::is_whitespace, "_")
}

/// One VTK cell: its type code and the point indices it spans.
struct VtkCell {
    cell_type: u8,
    points: Vec<usize>,
}

/// Geometry flattened for VTK: deduplicated points (with the node id behind
/// each, for field lookups) and the cells over them.
struct Geometry {
    points: Vec<[f64; 3]>,
    point_nodes: Vec<NodeId>,
    cells: Vec<VtkCell>,
}

/// Flatten a mesh into VTK points and cells (submesh by submesh, in order).
fn geometry(mesh: &Mesh) -> Result<Geometry> {
    let coords_h = mesh.coords()?;
    let coords = read(&coords_h)?;
    let dim = (coords.dim() as usize).min(3);

    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut point_nodes: Vec<NodeId> = Vec::new();
    let mut index: HashMap<NodeId, usize> = HashMap::new();
    let mut cells: Vec<VtkCell> = Vec::new();

    for sm_h in mesh {
        let sm = read(sm_h)?;
        let et = sm.element_type();
        let npc = et.nodes_per_cell();
        let cell_type = vtk_cell_type(et);
        for chunk in sm.connectivity().chunks(npc) {
            let mut pts = Vec::with_capacity(npc);
            for &nid in chunk {
                let idx = match index.get(&nid) {
                    Some(&i) => i,
                    None => {
                        let c = coords.coord(nid)?;
                        let mut p = [0.0; 3];
                        p[..dim].copy_from_slice(&c[..dim]);
                        let i = points.len();
                        points.push(p);
                        point_nodes.push(nid);
                        index.insert(nid, i);
                        i
                    }
                };
                pts.push(idx);
            }
            cells.push(VtkCell {
                cell_type,
                points: pts,
            });
        }
    }
    Ok(Geometry {
        points,
        point_nodes,
        cells,
    })
}

/// Write the header + geometry (`POINTS` / `CELLS` / `CELL_TYPES`).
fn write_geometry(out: &mut String, geo: &Geometry, title: &str) {
    out.push_str("# vtk DataFile Version 3.0\n");
    out.push_str(title);
    out.push('\n');
    out.push_str("ASCII\nDATASET UNSTRUCTURED_GRID\n");

    // POINTS / CELLS / CELL_TYPES are pure formatting of `geo` (no store
    // access): format each line in parallel, then append in order — byte-for-byte
    // identical to the sequential output.
    let _ = writeln!(out, "POINTS {} double", geo.points.len());
    let pts: Vec<String> = geo
        .points
        .par_iter()
        .with_min_len(MIN_PARALLEL_LEN)
        .map(|p| format!("{} {} {}\n", p[0], p[1], p[2]))
        .collect();
    pts.iter().for_each(|s| out.push_str(s));

    let conn_size: usize = geo.cells.iter().map(|c| 1 + c.points.len()).sum();
    let _ = writeln!(out, "CELLS {} {}", geo.cells.len(), conn_size);
    let cell_lines: Vec<String> = geo
        .cells
        .par_iter()
        .with_min_len(MIN_PARALLEL_LEN)
        .map(|c| {
            let mut line = c.points.len().to_string();
            for &i in &c.points {
                let _ = write!(line, " {i}");
            }
            line.push('\n');
            line
        })
        .collect();
    cell_lines.iter().for_each(|s| out.push_str(s));

    let _ = writeln!(out, "CELL_TYPES {}", geo.cells.len());
    let types: Vec<String> = geo
        .cells
        .par_iter()
        .with_min_len(MIN_PARALLEL_LEN)
        .map(|c| format!("{}\n", c.cell_type))
        .collect();
    types.iter().for_each(|s| out.push_str(s));
}

// ─── String builders (pure: no file I/O) ─────────────────────────────────────

/// Legacy-VTK text for a mesh (geometry only).
pub fn vtk_mesh_string(mesh: &Mesh) -> Result<String> {
    let geo = geometry(mesh)?;
    let mut out = String::new();
    write_geometry(&mut out, &geo, "pyrucast mesh");
    Ok(out)
}

/// Legacy-VTK text for `mesh` carrying `field` as `POINT_DATA`.
pub fn vtk_node_field_string(mesh: &Mesh, field: &NodeField) -> Result<String> {
    let geo = geometry(mesh)?;
    let mut out = String::new();
    write_geometry(&mut out, &geo, "pyrucast node field");

    let _ = writeln!(out, "POINT_DATA {}", geo.points.len());
    for comp in field.components()? {
        let _ = writeln!(out, "SCALARS {} double 1", sanitize(&comp));
        out.push_str("LOOKUP_TABLE default\n");
        for &nid in &geo.point_nodes {
            let v = field.value_opt(nid, &comp)?.unwrap_or(0.0);
            let _ = writeln!(out, "{v}");
        }
    }
    Ok(out)
}

/// Legacy-VTK text for `mesh` carrying `field` as `CELL_DATA`.
///
/// The cells are written submesh by submesh, in the mesh's order (matching the
/// geometry writer). A field zone is resolved from the submesh through its FE
/// support; several zones may share a support (they carry disjoint components,
/// per the union invariant), so the value for a `(submesh, component)` comes
/// from the **unique** zone on that support carrying the component — the field
/// must not fold cells across zones.
pub fn vtk_element_field_string(mesh: &Mesh, field: &ElementField) -> Result<String> {
    let geo = geometry(mesh)?;

    // Zones grouped by their support's submesh, so a submesh can be served by
    // several zones carrying disjoint components.
    let zones: Vec<Handle<crate::containers::element_field::SubElementField>> =
        field.iter().cloned().collect();
    // The value of `component` on `submesh`'s cells, or `None` if no zone on
    // that submesh carries it. Errors on a duplicate `(submesh, component)`.
    let cell_value =
        |submesh: &Handle<SubMesh>, component: &str, cell: usize| -> Result<Option<f64>> {
            let mut found: Option<f64> = None;
            for z in &zones {
                let sub = read(z)?;
                let sm = read(&sub.support())?.submesh();
                if sm.index() != submesh.index() || sm.generation() != submesh.generation() {
                    continue;
                }
                if !sub.components().iter().any(|c| c == component) {
                    continue;
                }
                if found.is_some() {
                    return Err(PyrucastError::Message(format!(
                        "vtk: component {component} is carried by two zones on the \
                     same support — consolidate the field first"
                    )));
                }
                let ng = sub.gauss_count();
                let v = if ng > 0 {
                    let mut acc = 0.0;
                    for g in 0..ng {
                        acc += sub.value(cell, g, component)?;
                    }
                    acc / ng as f64
                } else {
                    0.0
                };
                found = Some(v);
            }
            Ok(found)
        };

    // The mesh cells must line up with a zone's cells, submesh by submesh:
    // every mesh submesh must be covered by a zone built on it, with a matching
    // cell count. A field from a space built on a *different* mesh leaves some
    // submesh uncovered → error.
    for sm_h in mesh {
        let submesh_cells = read(sm_h)?.cell_count();
        let mut covered = false;
        for z in &zones {
            let sub = read(z)?;
            let sm = read(&sub.support())?.submesh();
            if sm.index() == sm_h.index() && sm.generation() == sm_h.generation() {
                if sub.cell_count() != submesh_cells {
                    return Err(PyrucastError::Message(format!(
                        "vtk: element field has {} cell(s) on a submesh with {} — \
                         the field must come from a space built on this mesh",
                        sub.cell_count(),
                        submesh_cells
                    )));
                }
                covered = true;
                break;
            }
        }
        if !covered {
            return Err(PyrucastError::Message(
                "vtk: a mesh submesh carries no element-field zone — \
                 the field must come from a space built on this mesh"
                    .into(),
            ));
        }
    }

    let mut out = String::new();
    write_geometry(&mut out, &geo, "pyrucast element field");

    let _ = writeln!(out, "CELL_DATA {}", geo.cells.len());
    for comp in field.components()? {
        let _ = writeln!(out, "SCALARS {} double 1", sanitize(&comp));
        out.push_str("LOOKUP_TABLE default\n");
        for sm_h in mesh {
            let n = read(sm_h)?.cell_count();
            for cell in 0..n {
                let v = cell_value(sm_h, &comp, cell)?.unwrap_or(0.0);
                let _ = writeln!(out, "{v}");
            }
        }
    }
    Ok(out)
}

// ─── File writers ────────────────────────────────────────────────────────────

/// Write a mesh (geometry only) to a legacy `.vtk` file.
pub fn write_vtk_mesh(mesh: &Mesh, path: &Path) -> Result<()> {
    std::fs::write(path, vtk_mesh_string(mesh)?)?;
    Ok(())
}

/// Write `mesh` + a [`NodeField`] (`POINT_DATA`) to a legacy `.vtk` file.
pub fn write_vtk_node_field(mesh: &Mesh, field: &NodeField, path: &Path) -> Result<()> {
    std::fs::write(path, vtk_node_field_string(mesh, field)?)?;
    Ok(())
}

/// Write `mesh` + an [`ElementField`] (`CELL_DATA`) to a legacy `.vtk` file.
pub fn write_vtk_element_field(mesh: &Mesh, field: &ElementField, path: &Path) -> Result<()> {
    std::fs::write(path, vtk_element_field_string(mesh, field)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Coords, Node, SubMesh};
    use crate::store::insert;

    /// Unit square as two TRI3 on a 2-D Coords, plus the four nodes.
    fn square() -> (Mesh, Vec<Node>) {
        let coords = insert(Coords::new(2).unwrap());
        let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
        sm.add_cell(&[n[0].id(), n[2].id(), n[3].id()]).unwrap();
        (Mesh::from_submesh(sm), n)
    }

    #[test]
    fn mesh_geometry_blocks() {
        let (mesh, _n) = square();
        let s = vtk_mesh_string(&mesh).unwrap();
        assert!(s.starts_with("# vtk DataFile Version 3.0\n"));
        assert!(s.contains("DATASET UNSTRUCTURED_GRID"));
        assert!(s.contains("POINTS 4 double"));
        // 2-D coords padded to 3-D.
        assert!(s.contains("0 0 0"));
        assert!(s.contains("1 1 0"));
        // 2 triangles → CELLS 2 8 (each "3 i j k"), both TRIANGLE (type 5).
        assert!(s.contains("CELLS 2 8"));
        assert!(s.contains("CELL_TYPES 2"));
        let fives = s.matches("\n5\n").count() + usize::from(s.ends_with("5\n"));
        assert!(fives >= 2);
    }

    #[test]
    fn node_field_point_data() {
        use crate::containers::node_field::SubNodeField;
        let (mesh, n) = square();
        let support = insert(SubMesh::poi1_from_nodes(&n).unwrap());
        let mut sub = SubNodeField::from_poi1(&support, vec!["T".into()]).unwrap();
        for (i, node) in n.iter().enumerate() {
            sub.set_value(node.id(), "T", i as f64 * 10.0).unwrap();
        }
        let field = NodeField::from_sub(sub);
        let s = vtk_node_field_string(&mesh, &field).unwrap();
        assert!(s.contains("POINT_DATA 4"));
        assert!(s.contains("SCALARS T double 1"));
        assert!(s.contains("LOOKUP_TABLE default"));
        // Values appear in point order: 0, 10, 20, 30.
        for v in ["0\n", "10\n", "20\n", "30\n"] {
            assert!(s.contains(v), "missing {v:?}");
        }
    }

    #[test]
    fn element_field_cell_data_is_gauss_mean() {
        let (mesh, _n) = square();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let field = ElementField::new(&fes, vec!["s".into()]).unwrap();
        // Set every Gauss point of cell 0 to 2.0 and cell 1 to 5.0.
        for sub_h in field.iter() {
            // single sub for a single-type mesh
            let mut sub = crate::store::write(sub_h).unwrap();
            let ng = sub.gauss_count();
            for g in 0..ng {
                sub.set_value(0, g, "s", 2.0).unwrap();
                sub.set_value(1, g, "s", 5.0).unwrap();
            }
        }
        let s = vtk_element_field_string(&mesh, &field).unwrap();
        assert!(s.contains("CELL_DATA 2"));
        assert!(s.contains("SCALARS s double 1"));
        assert!(s.contains("2\n"));
        assert!(s.contains("5\n"));
    }

    #[test]
    fn element_field_cell_mismatch_errors() {
        let (mesh, n) = square();
        // A field built on a one-triangle mesh (same Coords) has 1 cell,
        // while `mesh` has 2 → exporting the field against `mesh` must error.
        let mut sm = SubMesh::new(mesh.coords().unwrap(), ElementType::TRI3);
        sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
        let other = Mesh::from_submesh(sm);
        let fes = FiniteElementSpace::lagrange1(&other).unwrap();
        let field = ElementField::new(&fes, vec!["s".into()]).unwrap();
        assert!(vtk_element_field_string(&mesh, &field).is_err());
    }
}
