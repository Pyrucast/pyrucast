//! Export a mesh or a field to a **legacy VTK** file (`UNSTRUCTURED_GRID`)
//! for viewing in ParaView — in **ASCII** or **binary**, one field per file,
//! or an [`Evolution`] as a **time series** ParaView plays back.
//!
//! The legacy `.vtk` format is the simplest ParaView reads natively: a
//! header, then `POINTS` / `CELLS` / `CELL_TYPES`, then optional
//! `POINT_DATA` (a [`NodeField`]) or `CELL_DATA` (an [`ElementField`]).
//! This module is only a formatter: the layout — which points, which cells in
//! which order, which values — comes from
//! [`to_arrays`], the exit every
//! exchange format shares.
//!
//! - **Geometry.** Every submesh of the [`Mesh`] is written; the nodes it
//!   references become VTK points, in order of first appearance, padded to
//!   3-D with `z = 0` for a 2-D `Coords`. Cells come type by type, in order of
//!   first appearance. Cell types map one-to-one and the local node ordering
//!   already matches VTK's, so connectivity is copied verbatim:
//!
//!   | [`ElementType`](crate::atoms::ElementType) | VTK cell | code |
//!   |---|---|---|
//!   | `POI1` | `VERTEX`       | 1  |
//!   | `SEG2` | `LINE`         | 3  |
//!   | `TRI3` | `TRIANGLE`     | 5  |
//!   | `QUA4` | `QUAD`         | 9  |
//!   | `TET4` | `TETRA`        | 10 |
//!   | `PYRA5` | `PYRAMID`     | 14 |
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
//!   cell). The field must cover every cell of the mesh: it comes from a
//!   space built on the **same** mesh.
//! - **Binary** ([`VtkEncoding::Binary`]): the same sections, the numbers
//!   written raw in **big-endian** as the legacy format requires — much
//!   smaller and faster to read for a large mesh.
//! - **Time series** ([`write_vtk_series`]): one file per tabulated value of
//!   an [`Evolution`] of fields, plus a `.vtk.series` index giving each file
//!   its time — ParaView opens the index as one dataset with a time slider.
//!   The mesh is laid out once for all the frames.

use crate::containers::element_field::ElementField;
use crate::containers::evolution::{Evolution, ValueKind};
use crate::containers::mesh::Mesh;
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};
use crate::ops::export::arrays::{to_arrays, ElementLayout, Exported};
use crate::ops::mesh::arrays::NodeOrder;
use crate::parallel::*;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// How the numbers of a legacy VTK file are written.
///
/// ```
/// # use pyrucast::ops::export::vtk::VtkEncoding;
/// assert_ne!(VtkEncoding::Ascii, VtkEncoding::Binary);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VtkEncoding {
    /// Text: readable, diffable, large.
    Ascii,
    /// Raw big-endian numbers: compact and fast.
    Binary,
}

/// VTK array names cannot carry spaces; swap them for underscores.
fn sanitize(name: &str) -> String {
    name.replace(char::is_whitespace, "_")
}

/// What a file carries besides the geometry.
enum Data<'a> {
    None,
    Node(&'a NodeField),
    Element(&'a ElementField),
}

/// The byte sink of one file: text sections, then numbers in the chosen
/// encoding.
struct Sink {
    out: Vec<u8>,
    encoding: VtkEncoding,
}

impl Sink {
    fn text(&mut self, s: &str) {
        self.out.extend_from_slice(s.as_bytes());
    }

    /// Numbers, one per line in ASCII (formatted in parallel, appended in
    /// order: byte-for-byte the sequential output), raw big-endian otherwise.
    fn f64s(&mut self, values: &[f64]) {
        match self.encoding {
            VtkEncoding::Ascii => {
                let lines: Vec<String> = values
                    .par_iter()
                    .with_min_len(MIN_PARALLEL_LEN)
                    .map(|v| format!("{v}\n"))
                    .collect();
                lines.iter().for_each(|s| self.text(s));
            }
            VtkEncoding::Binary => {
                self.out.reserve(8 * values.len() + 1);
                for v in values {
                    self.out.extend_from_slice(&v.to_be_bytes());
                }
                self.out.push(b'\n');
            }
        }
    }
}

/// Header + `POINTS` / `CELLS` / `CELL_TYPES` of a layout.
fn write_geometry(sink: &mut Sink, arrays: &Exported, title: &str) {
    sink.text("# vtk DataFile Version 3.0\n");
    sink.text(title);
    sink.text(match sink.encoding {
        VtkEncoding::Ascii => "\nASCII\nDATASET UNSTRUCTURED_GRID\n",
        VtkEncoding::Binary => "\nBINARY\nDATASET UNSTRUCTURED_GRID\n",
    });

    let d = arrays.dim.min(3);
    let n = arrays.node_tags.len();
    let point = |k: usize| {
        let mut p = [0.0; 3];
        p[..d].copy_from_slice(&arrays.node_coords[k * arrays.dim..k * arrays.dim + d]);
        p
    };
    sink.text(&format!("POINTS {n} double\n"));
    match sink.encoding {
        VtkEncoding::Ascii => {
            let lines: Vec<String> = (0..n)
                .into_par_iter()
                .with_min_len(MIN_PARALLEL_LEN)
                .map(|k| {
                    let p = point(k);
                    format!("{} {} {}\n", p[0], p[1], p[2])
                })
                .collect();
            lines.iter().for_each(|s| sink.text(s));
        }
        VtkEncoding::Binary => {
            sink.out.reserve(24 * n + 1);
            for k in 0..n {
                for c in point(k) {
                    sink.out.extend_from_slice(&c.to_be_bytes());
                }
            }
            sink.out.push(b'\n');
        }
    }

    let n_cells: usize = arrays.blocks.iter().map(|b| b.cell_tags.len()).sum();
    let size: usize = arrays
        .blocks
        .iter()
        .map(|b| b.cell_tags.len() * (1 + b.element_type.nodes_per_cell()))
        .sum();
    sink.text(&format!("CELLS {n_cells} {size}\n"));
    match sink.encoding {
        VtkEncoding::Ascii => {
            for b in &arrays.blocks {
                let npc = b.element_type.nodes_per_cell();
                let lines: Vec<String> = b
                    .node_tags
                    .par_chunks(npc)
                    .with_min_len(MIN_PARALLEL_LEN)
                    .map(|cell| {
                        let mut line = npc.to_string();
                        for &i in cell {
                            let _ = write!(line, " {i}");
                        }
                        line.push('\n');
                        line
                    })
                    .collect();
                lines.iter().for_each(|s| sink.text(s));
            }
        }
        VtkEncoding::Binary => {
            sink.out.reserve(4 * size + 1);
            for b in &arrays.blocks {
                let npc = b.element_type.nodes_per_cell();
                for cell in b.node_tags.chunks_exact(npc) {
                    sink.out.extend_from_slice(&(npc as i32).to_be_bytes());
                    for &i in cell {
                        sink.out.extend_from_slice(&(i as i32).to_be_bytes());
                    }
                }
            }
            sink.out.push(b'\n');
        }
    }

    sink.text(&format!("CELL_TYPES {n_cells}\n"));
    for b in &arrays.blocks {
        let code = b.element_type.as_kind().vtk_code();
        match sink.encoding {
            VtkEncoding::Ascii => {
                let line = format!("{code}\n");
                for _ in 0..b.cell_tags.len() {
                    sink.text(&line);
                }
            }
            VtkEncoding::Binary => {
                for _ in 0..b.cell_tags.len() {
                    sink.out.extend_from_slice(&i32::from(code).to_be_bytes());
                }
            }
        }
    }
    if sink.encoding == VtkEncoding::Binary {
        sink.out.push(b'\n');
    }
}

/// `POINT_DATA`: one `SCALARS` per component of node field `f`.
fn write_point_data(sink: &mut Sink, arrays: &Exported, f: usize) {
    let field = &arrays.node_fields[f];
    let n = arrays.node_tags.len();
    let nc = field.components.len();
    sink.text(&format!("POINT_DATA {n}\n"));
    let mut column = Vec::with_capacity(n);
    for (c, name) in field.components.iter().enumerate() {
        sink.text(&format!(
            "SCALARS {} double 1\nLOOKUP_TABLE default\n",
            sanitize(name)
        ));
        column.clear();
        column.extend(field.values.iter().skip(c).step_by(nc.max(1)));
        sink.f64s(&column);
    }
}

/// `CELL_DATA`: one `SCALARS` per component of cell field `f`, which must
/// cover every cell (the export tags are `0..n_cells`, block after block).
fn write_cell_data(sink: &mut Sink, arrays: &Exported, f: usize) -> Result<()> {
    let field = &arrays.cell_fields[f];
    let n_cells: usize = arrays.blocks.iter().map(|b| b.cell_tags.len()).sum();
    if field.cell_tags.len() != n_cells {
        return Err(PyrucastError::Message(
            "vtk: a mesh submesh carries no element-field zone — \
             the field must come from a space built on this mesh"
                .into(),
        ));
    }
    let nc = field.components.len();
    // The cells come back zone by zone; VTK wants them in cell order.
    let mut at = vec![0usize; n_cells];
    for (k, &t) in field.cell_tags.iter().enumerate() {
        at[t as usize] = k;
    }
    sink.text(&format!("CELL_DATA {n_cells}\n"));
    let mut column = vec![0.0; n_cells];
    for (c, name) in field.components.iter().enumerate() {
        sink.text(&format!(
            "SCALARS {} double 1\nLOOKUP_TABLE default\n",
            sanitize(name)
        ));
        for (dst, &k) in column.iter_mut().zip(&at) {
            *dst = field.values[k * nc + c];
        }
        sink.f64s(&column);
    }
    Ok(())
}

/// Lay `mesh` (and the fields) out once, in pyrucast's order with tags from
/// zero — VTK point and cell indices.
fn layout(mesh: &Mesh, nodes: &[&NodeField], elements: &[&ElementField]) -> Result<Exported> {
    let elements: Vec<_> = elements.iter().map(|&f| (f, ElementLayout::Cell)).collect();
    to_arrays(
        &[(String::new(), mesh)],
        nodes,
        &elements,
        NodeOrder::Pyrucast,
        0,
    )
}

/// The bytes of one legacy VTK file for `mesh`, carrying `data`.
fn vtk_bytes(mesh: &Mesh, data: Data<'_>, encoding: VtkEncoding) -> Result<Vec<u8>> {
    let (arrays, title) = match data {
        Data::None => (layout(mesh, &[], &[])?, "pyrucast mesh"),
        Data::Node(f) => (layout(mesh, &[f], &[])?, "pyrucast node field"),
        Data::Element(f) => (layout(mesh, &[], &[f])?, "pyrucast element field"),
    };
    let mut sink = Sink {
        out: Vec::new(),
        encoding,
    };
    write_geometry(&mut sink, &arrays, title);
    match data {
        Data::None => {}
        Data::Node(_) => write_point_data(&mut sink, &arrays, 0),
        Data::Element(_) => write_cell_data(&mut sink, &arrays, 0)?,
    }
    Ok(sink.out)
}

fn ascii(bytes: Vec<u8>) -> String {
    String::from_utf8(bytes).expect("an ASCII VTK file is made of `format!` output only")
}

// ─── String builders (ASCII, pure: no file I/O) ──────────────────────────────

/// Legacy-VTK text for a mesh (geometry only).
///
/// ```
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::export;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// // Le VTK « legacy », en texte : points, cellules, types de cellules.
/// let s = export::vtk::vtk_mesh_string(&maillage)?;
/// assert!(s.starts_with("# vtk DataFile Version"));
/// assert!(s.contains("POINTS 3 double"));
/// assert!(s.contains("CELL_TYPES 1"));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn vtk_mesh_string(mesh: &Mesh) -> Result<String> {
    Ok(ascii(vtk_bytes(mesh, Data::None, VtkEncoding::Ascii)?))
}

/// Legacy-VTK text for `mesh` carrying `field` as `POINT_DATA`.
///
/// ```
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{export, mesh as ops_mesh};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let support = ops_mesh::poi1_from_nodes(&n).unwrap();
/// # let temp = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
/// # temp.get(0).unwrap().write().add_to_component("T", 20.0).unwrap();
/// // The same mesh, plus the values **at the nodes**.
/// let s = export::vtk::vtk_node_field_string(&maillage, &temp)?;
/// assert!(s.contains("POINT_DATA 3"));
/// assert!(s.contains("SCALARS T double 1"));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn vtk_node_field_string(mesh: &Mesh, field: &NodeField) -> Result<String> {
    Ok(ascii(vtk_bytes(
        mesh,
        Data::Node(field),
        VtkEncoding::Ascii,
    )?))
}

/// Legacy-VTK text for `mesh` carrying `field` as `CELL_DATA` — the Gauss
/// mean per cell. The field must cover every cell of `mesh`; several zones
/// may share a support (they carry disjoint components, per the union
/// invariant), but a component carried twice on one support is refused.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::export;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let mut flux = ElementField::new(&fes, vec!["q".into()]).unwrap();
/// # flux.get(0).unwrap().write().set_uniform("q", 1.0).unwrap();
/// // And here the values **per cell**: the Gauss points are averaged per
/// // cell, since VTK knows no data at the integration point.
/// let s = export::vtk::vtk_element_field_string(&maillage, &flux)?;
/// assert!(s.contains("CELL_DATA 1"));
/// assert!(s.contains("SCALARS q double 1"));
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn vtk_element_field_string(mesh: &Mesh, field: &ElementField) -> Result<String> {
    Ok(ascii(vtk_bytes(
        mesh,
        Data::Element(field),
        VtkEncoding::Ascii,
    )?))
}

// ─── File writers ────────────────────────────────────────────────────────────

/// Write a mesh (geometry only) to a legacy `.vtk` file.
///
/// ```
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::export::{self, vtk::VtkEncoding};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let dossier = std::env::temp_dir()
/// #     .join(format!("pyrucast_vtk_mesh_{}", std::process::id()));
/// # std::fs::create_dir_all(&dossier).unwrap();
/// let chemin = dossier.join("maillage.vtk");
/// export::vtk::write_vtk_mesh(&maillage, &chemin, VtkEncoding::Ascii)?;
/// assert_eq!(std::fs::read_to_string(&chemin)?,
///            export::vtk::vtk_mesh_string(&maillage)?);
/// // Binary: same sections, raw big-endian numbers.
/// export::vtk::write_vtk_mesh(&maillage, &chemin, VtkEncoding::Binary)?;
/// assert!(std::fs::read(&chemin)?.windows(6).any(|w| w == b"BINARY"));
/// # let _ = std::fs::remove_dir_all(&dossier);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn write_vtk_mesh(mesh: &Mesh, path: &Path, encoding: VtkEncoding) -> Result<()> {
    std::fs::write(path, vtk_bytes(mesh, Data::None, encoding)?)?;
    Ok(())
}

/// Write `mesh` + a [`NodeField`] (`POINT_DATA`) to a legacy `.vtk` file.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{export::{self, vtk::VtkEncoding}, mesh as ops_mesh};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let support = ops_mesh::poi1_from_nodes(&n).unwrap();
/// # let temp = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
/// # let dossier = std::env::temp_dir()
/// #     .join(format!("pyrucast_vtk_node_{}", std::process::id()));
/// # std::fs::create_dir_all(&dossier).unwrap();
/// let chemin = dossier.join("temperature.vtk");
/// export::vtk::write_vtk_node_field(&maillage, &temp, &chemin, VtkEncoding::Ascii)?;
/// assert!(std::fs::read_to_string(&chemin)?.contains("POINT_DATA 3"));
/// # let _ = std::fs::remove_dir_all(&dossier);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn write_vtk_node_field(
    mesh: &Mesh,
    field: &NodeField,
    path: &Path,
    encoding: VtkEncoding,
) -> Result<()> {
    std::fs::write(path, vtk_bytes(mesh, Data::Node(field), encoding)?)?;
    Ok(())
}

/// Write `mesh` + an [`ElementField`] (`CELL_DATA`) to a legacy `.vtk` file.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::export::{self, vtk::VtkEncoding};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let mut flux = ElementField::new(&fes, vec!["q".into()]).unwrap();
/// # flux.get(0).unwrap().write().set_uniform("q", 1.0).unwrap();
/// # let dossier = std::env::temp_dir()
/// #     .join(format!("pyrucast_vtk_elem_{}", std::process::id()));
/// # std::fs::create_dir_all(&dossier).unwrap();
/// let chemin = dossier.join("flux.vtk");
/// export::vtk::write_vtk_element_field(&maillage, &flux, &chemin, VtkEncoding::Ascii)?;
/// assert!(std::fs::read_to_string(&chemin)?.contains("CELL_DATA 1"));
/// # let _ = std::fs::remove_dir_all(&dossier);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn write_vtk_element_field(
    mesh: &Mesh,
    field: &ElementField,
    path: &Path,
    encoding: VtkEncoding,
) -> Result<()> {
    std::fs::write(path, vtk_bytes(mesh, Data::Element(field), encoding)?)?;
    Ok(())
}

/// Write an [`Evolution`] of node or element fields as a **VTK time
/// series**: one legacy file per tabulated value, `stem_0000.vtk`,
/// `stem_0001.vtk`, … next to `path`, plus the index `path` itself — a
/// `.vtk.series` JSON file giving each file its abscissa as time. ParaView
/// opens the index as one dataset with a time slider.
///
/// The mesh is laid out once for all the frames. Returns the paths written,
/// the index last. Errors if the evolution tabulates scalars, or if its
/// zones do not share their abscissas.
///
/// ```
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::evolution::{Evolution, OutOfRange};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{export::{self, vtk::VtkEncoding}, mesh as ops_mesh};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let support = ops_mesh::poi1_from_nodes(&n).unwrap();
/// # let froid = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
/// # let chaud = NodeField::from_submesh(&support.get(0).unwrap(), vec!["T".into()]).unwrap();
/// # chaud.get(0).unwrap().write().add_to_component("T", 100.0).unwrap();
/// # let dossier = std::env::temp_dir()
/// #     .join(format!("pyrucast_vtk_series_{}", std::process::id()));
/// # std::fs::create_dir_all(&dossier).unwrap();
/// let t = Evolution::from_node_fields(&[(0.0, &froid), (2.5, &chaud)], OutOfRange::Error)?;
/// let index = dossier.join("chauffe.vtk.series");
/// let written = export::vtk::write_vtk_series(&maillage, &t, &index, VtkEncoding::Binary)?;
/// assert_eq!(written.len(), 3); // two frames, then the index
/// let json = std::fs::read_to_string(&index)?;
/// assert!(json.contains("\"name\": \"chauffe_0001.vtk\", \"time\": 2.5"));
/// # let _ = std::fs::remove_dir_all(&dossier);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn write_vtk_series(
    mesh: &Mesh,
    evolution: &Evolution,
    path: &Path,
    encoding: VtkEncoding,
) -> Result<Vec<PathBuf>> {
    let times = evolution.shared_abscissas()?;
    let kind = evolution.kind()?;
    let (arrays, title) = match kind {
        ValueKind::Node => {
            let frames = (0..times.len())
                .map(|k| evolution.node_frame(k))
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<&NodeField> = frames.iter().collect();
            (layout(mesh, &refs, &[])?, "pyrucast node field")
        }
        ValueKind::Element => {
            let frames = (0..times.len())
                .map(|k| evolution.element_frame(k))
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<&ElementField> = frames.iter().collect();
            (layout(mesh, &[], &refs)?, "pyrucast element field")
        }
        ValueKind::Scalar => {
            return Err(PyrucastError::Message(
                "vtk: a series needs an evolution of fields, not of scalars".into(),
            ));
        }
    };

    // `stem` from `stem.vtk.series` (or any other name, minus its extensions).
    let dir = path.parent().unwrap_or(Path::new(""));
    let file = path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("series");
    let stem = file.split('.').next().unwrap_or(file);

    let mut geometry = Sink {
        out: Vec::new(),
        encoding,
    };
    write_geometry(&mut geometry, &arrays, title);
    let mut written = Vec::with_capacity(times.len() + 1);
    let mut index = String::from("{\n  \"file-series-version\": \"1.0\",\n  \"files\": [\n");
    for (k, &t) in times.iter().enumerate() {
        let name = format!("{stem}_{k:04}.vtk");
        let mut sink = Sink {
            out: geometry.out.clone(),
            encoding,
        };
        match kind {
            ValueKind::Node => write_point_data(&mut sink, &arrays, k),
            _ => write_cell_data(&mut sink, &arrays, k)?,
        }
        let target = dir.join(&name);
        std::fs::write(&target, sink.out)?;
        written.push(target);
        let sep = if k + 1 < times.len() { "," } else { "" };
        let _ = writeln!(index, "    {{ \"name\": \"{name}\", \"time\": {t} }}{sep}");
    }
    index.push_str("  ]\n}\n");
    std::fs::write(path, index)?;
    written.push(path.to_path_buf());
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::ElementType;
    use crate::atoms::Node;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

    /// Unit square as two TRI3 on a 2-D Coords, plus the four nodes.
    fn square() -> (Mesh, Vec<Node>) {
        let coords = Handle::new(Coords::new(2).unwrap());
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
        let support = Handle::new(SubMesh::poi1_from_nodes(&n).unwrap());
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
            let mut sub = sub_h.write();
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

#[cfg(test)]
mod encoding_tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node};
    use crate::containers::evolution::OutOfRange;
    use crate::containers::field::SubField;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;
    use crate::handle::Handle;

    fn square() -> Mesh {
        let coords = Handle::new(Coords::new(2).unwrap());
        let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            .iter()
            .map(|c| Node::create_in(coords.clone(), c).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
        sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
        sm.add_cell(&[n[0].id(), n[2].id(), n[3].id()]).unwrap();
        Mesh::from_submesh(sm)
    }

    /// The binary file holds the ASCII file's numbers, raw and big-endian.
    #[test]
    fn binary_carries_the_same_numbers() {
        let mesh = square();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let field = ElementField::new(&fes, vec!["s".into()]).unwrap();
        field.get(0).unwrap().write().set_uniform("s", 2.5).unwrap();
        let bin = vtk_bytes(&mesh, Data::Element(&field), VtkEncoding::Binary).unwrap();
        let find = |pat: &[u8]| bin.windows(pat.len()).position(|w| w == pat).unwrap() + pat.len();
        assert!(bin.windows(6).any(|w| w == b"BINARY"));
        // POINTS: 4 × 3 doubles; the third point is (1, 1, 0).
        let at = find(b"POINTS 4 double\n");
        let third: Vec<f64> = (0..3)
            .map(|k| f64::from_be_bytes(bin[at + 48 + 8 * k..at + 56 + 8 * k].try_into().unwrap()))
            .collect();
        assert_eq!(third, [1.0, 1.0, 0.0]);
        // CELLS: "3 0 1 2" as big-endian i32.
        let at = find(b"CELLS 2 8\n");
        let first: Vec<i32> = (0..4)
            .map(|k| i32::from_be_bytes(bin[at + 4 * k..at + 4 * k + 4].try_into().unwrap()))
            .collect();
        assert_eq!(first, [3, 0, 1, 2]);
        let at = find(b"LOOKUP_TABLE default\n");
        assert_eq!(f64::from_be_bytes(bin[at..at + 8].try_into().unwrap()), 2.5);
    }

    #[test]
    fn a_series_writes_one_file_per_frame_and_an_index() {
        let mesh = square();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let cold = ElementField::new(&fes, vec!["s".into()]).unwrap();
        let hot = ElementField::new(&fes, vec!["s".into()]).unwrap();
        hot.get(0).unwrap().write().set_uniform("s", 9.0).unwrap();
        let e = Evolution::from_element_fields(&[(0.0, &cold), (0.5, &hot)], OutOfRange::Error)
            .unwrap();
        let dir = std::env::temp_dir().join(format!("pyrucast_series_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let index = dir.join("run.vtk.series");
        let written = write_vtk_series(&mesh, &e, &index, VtkEncoding::Ascii).unwrap();
        assert_eq!(
            written,
            [
                dir.join("run_0000.vtk"),
                dir.join("run_0001.vtk"),
                index.clone()
            ]
        );
        let last = std::fs::read_to_string(&written[1]).unwrap();
        assert!(last.contains("CELL_DATA 2\nSCALARS s double 1\nLOOKUP_TABLE default\n9\n9\n"));
        let json = std::fs::read_to_string(&index).unwrap();
        assert!(json.contains("{ \"name\": \"run_0000.vtk\", \"time\": 0 },"));
        assert!(json.contains("{ \"name\": \"run_0001.vtk\", \"time\": 0.5 }\n"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
