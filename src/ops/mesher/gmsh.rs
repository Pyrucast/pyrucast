//! Read a gmsh `.msh` mesh (ASCII) into pyrucast meshes.
//!
//! Supported on-disk formats: **MSH 2.2** and **MSH 4.1**, both in their
//! ASCII variant (`$MeshFormat … 0 …`); binary files are rejected with a
//! clear message. The element types pyrucast knows are mapped from their
//! gmsh codes:
//!
//! | gmsh code | [`ElementType`] |
//! |---|---|
//! | `1`  | `SEG2` |
//! | `2`  | `TRI3` |
//! | `3`  | `QUA4` |
//! | `4`  | `TET4` |
//! | `5`  | `HEX8` |
//! | `15` | `POI1` |
//!
//! Any other gmsh element type (higher-order, prism, pyramid, …) is an
//! error. The local node ordering of every supported type already matches
//! pyrucast's reference frame (see
//! [`crate::containers::mesh::ElementType`]), so the connectivity is copied
//! verbatim — no reordering.
//!
//! # Grouping
//!
//! The result is **one [`Mesh`] per gmsh physical group**, returned as a
//! list of `(group name, Mesh)` pairs in order of first appearance. Inside
//! a group's mesh there is **one [`SubMesh`] per element type**. All meshes
//! share a **single [`Coords`]**, so a node shared between two groups (e.g.
//! a boundary node belonging both to a surface and to its bounding line) is
//! the same node on both sides — handy for posing boundary conditions on a
//! named region read straight from the file. The Python binding turns the
//! list into a `dict[str, Mesh]`.
//!
//! Elements that carry no physical group land under the synthetic name
//! `"<ungrouped>"`. Only nodes actually **referenced** by an element are
//! materialized in the `Coords`; isolated nodes listed in the file but used
//! by no element are skipped.
//!
//! # Dimension
//!
//! The caller supplies the `Coords` to read into, so its dimension decides
//! how many coordinates are kept: gmsh always stores three per node, of
//! which the first `coords.dim()` are taken (a 2-D `Coords` flattens the
//! mesh onto its `xy` projection). The `Coords` may already hold geometry —
//! the import is merged into it.

use crate::aggregate::Aggregate;
use crate::containers::mesh::{Coords, ElementType, Mesh, Node, NodeId, SubMesh};
use crate::error::{PyrucastError, Result};
use crate::store::{insert, read, Handle};
use std::collections::HashMap;
use std::path::Path;

/// Synthetic group name for elements without a physical group.
const UNGROUPED: &str = "<ungrouped>";

fn err(msg: impl Into<String>) -> PyrucastError {
    PyrucastError::Message(msg.into())
}

/// Map a gmsh element-type code to a pyrucast [`ElementType`].
fn element_type_from_gmsh(code: u32) -> Result<ElementType> {
    Ok(match code {
        1 => ElementType::SEG2,
        2 => ElementType::TRI3,
        3 => ElementType::QUA4,
        4 => ElementType::TET4,
        5 => ElementType::HEX8,
        15 => ElementType::POI1,
        other => {
            return Err(err(format!(
                "gmsh: unsupported element type {other} (supported: 1=SEG2, \
                 2=TRI3, 3=QUA4, 4=TET4, 5=HEX8, 15=POI1)"
            )))
        }
    })
}

// ─── Parsed (pure) representation ────────────────────────────────────────────

/// One element as read from the file: its pyrucast type, the gmsh node tags
/// (in file order), and the physical-group names it belongs to.
struct ParsedElement {
    element_type: ElementType,
    groups: Vec<String>,
    nodes: Vec<u64>,
}

/// The pure result of parsing — coordinates by gmsh node tag and the list of
/// elements. No store access happens here.
struct Parsed {
    coords: HashMap<u64, [f64; 3]>,
    elements: Vec<ParsedElement>,
}

// ─── Token cursor over a section's lines ─────────────────────────────────────

/// Whitespace-token cursor over the (flattened) lines of one `$Section`.
/// gmsh sections are token streams whose structure is driven by leading
/// counts, so line boundaries don't matter once the counts are known.
struct Cursor<'a> {
    toks: Vec<&'a str>,
    i: usize,
}

impl<'a> Cursor<'a> {
    fn new(lines: &[&'a str]) -> Self {
        let toks = lines.iter().flat_map(|l| l.split_whitespace()).collect();
        Self { toks, i: 0 }
    }

    fn next_tok(&mut self) -> Result<&'a str> {
        let t = self
            .toks
            .get(self.i)
            .copied()
            .ok_or_else(|| err("gmsh: unexpected end of section"))?;
        self.i += 1;
        Ok(t)
    }

    fn u64(&mut self) -> Result<u64> {
        let t = self.next_tok()?;
        t.parse()
            .map_err(|_| err(format!("gmsh: expected an integer, got {t:?}")))
    }

    fn i64(&mut self) -> Result<i64> {
        let t = self.next_tok()?;
        t.parse()
            .map_err(|_| err(format!("gmsh: expected an integer, got {t:?}")))
    }

    fn usize(&mut self) -> Result<usize> {
        Ok(self.u64()? as usize)
    }

    fn f64(&mut self) -> Result<f64> {
        let t = self.next_tok()?;
        t.parse()
            .map_err(|_| err(format!("gmsh: expected a number, got {t:?}")))
    }
}

/// Split the file text into `name -> body lines`, one entry per `$Name …
/// $EndName` block. Unknown sections are kept too (and simply never read).
fn split_sections(text: &str) -> HashMap<String, Vec<&str>> {
    let mut map = HashMap::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let t = line.trim();
        let Some(name) = t.strip_prefix('$') else {
            continue;
        };
        if name.starts_with("End") {
            continue;
        }
        let end = format!("$End{name}");
        let mut body = Vec::new();
        for l in lines.by_ref() {
            if l.trim() == end {
                break;
            }
            body.push(l);
        }
        map.insert(name.to_string(), body);
    }
    map
}

/// Parse `$PhysicalNames` into `(dim, physical tag) -> name`.
fn parse_physical_names(map: &HashMap<String, Vec<&str>>) -> Result<HashMap<(u8, i64), String>> {
    let mut names = HashMap::new();
    let Some(lines) = map.get("PhysicalNames") else {
        return Ok(names);
    };
    let mut it = lines.iter().map(|l| l.trim()).filter(|l| !l.is_empty());
    let count: usize = it
        .next()
        .ok_or_else(|| err("gmsh: empty $PhysicalNames"))?
        .parse()
        .map_err(|_| err("gmsh: bad $PhysicalNames count"))?;
    for _ in 0..count {
        let line = it
            .next()
            .ok_or_else(|| err("gmsh: truncated $PhysicalNames"))?;
        let q1 = line
            .find('"')
            .ok_or_else(|| err("gmsh: $PhysicalNames entry without a quoted name"))?;
        let q2 = line
            .rfind('"')
            .filter(|&q| q > q1)
            .ok_or_else(|| err("gmsh: $PhysicalNames entry without a closing quote"))?;
        let name = line[q1 + 1..q2].to_string();
        let mut head = line[..q1].split_whitespace();
        let dim: u8 = head
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| err("gmsh: $PhysicalNames entry without a dimension"))?;
        let tag: i64 = head
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| err("gmsh: $PhysicalNames entry without a tag"))?;
        names.insert((dim, tag), name);
    }
    Ok(names)
}

/// Resolve a list of physical tags (of the given dimension) to group names,
/// falling back to `"<ungrouped>"` when there is none.
fn groups_of(dim: u8, phys: &[i64], names: &HashMap<(u8, i64), String>) -> Vec<String> {
    if phys.is_empty() {
        return vec![UNGROUPED.to_string()];
    }
    phys.iter()
        .map(|&p| {
            names
                .get(&(dim, p))
                .cloned()
                .unwrap_or_else(|| format!("physical {p}"))
        })
        .collect()
}

// ─── MSH 2.2 ─────────────────────────────────────────────────────────────────

fn parse_v2(
    map: &HashMap<String, Vec<&str>>,
    names: &HashMap<(u8, i64), String>,
) -> Result<Parsed> {
    let nodes_sec = map
        .get("Nodes")
        .ok_or_else(|| err("gmsh: missing $Nodes"))?;
    let mut c = Cursor::new(nodes_sec);
    let n = c.usize()?;
    let mut coords = HashMap::with_capacity(n);
    for _ in 0..n {
        let tag = c.u64()?;
        let xyz = [c.f64()?, c.f64()?, c.f64()?];
        coords.insert(tag, xyz);
    }

    let elems_sec = map
        .get("Elements")
        .ok_or_else(|| err("gmsh: missing $Elements"))?;
    let mut c = Cursor::new(elems_sec);
    let ne = c.usize()?;
    let mut elements = Vec::with_capacity(ne);
    for _ in 0..ne {
        let _id = c.u64()?;
        let et = element_type_from_gmsh(c.u64()? as u32)?;
        let ntags = c.usize()?;
        // In MSH 2.2 the first tag is the physical group, the second the
        // elementary (geometric) entity; the rest are partition tags.
        let mut phys: Vec<i64> = Vec::new();
        for k in 0..ntags {
            let t = c.i64()?;
            if k == 0 && t != 0 {
                phys.push(t);
            }
        }
        let npc = et.nodes_per_cell();
        let mut nodes = Vec::with_capacity(npc);
        for _ in 0..npc {
            nodes.push(c.u64()?);
        }
        let groups = groups_of(et.topological_dim() as u8, &phys, names);
        elements.push(ParsedElement {
            element_type: et,
            groups,
            nodes,
        });
    }
    Ok(Parsed { coords, elements })
}

// ─── MSH 4.1 ─────────────────────────────────────────────────────────────────

/// Parse `$Entities` into `(dim, entity tag) -> [physical tags]`.
fn parse_entities_v4(map: &HashMap<String, Vec<&str>>) -> Result<HashMap<(u8, i64), Vec<i64>>> {
    let mut entity_phys = HashMap::new();
    let Some(sec) = map.get("Entities") else {
        return Ok(entity_phys);
    };
    let mut c = Cursor::new(sec);
    let np = c.usize()?;
    let nc = c.usize()?;
    let ns = c.usize()?;
    let nv = c.usize()?;
    // Points: tag, 3 coords, numPhysical, physical tags.
    for _ in 0..np {
        let tag = c.i64()?;
        for _ in 0..3 {
            c.f64()?;
        }
        let k = c.usize()?;
        let mut phys = Vec::with_capacity(k);
        for _ in 0..k {
            phys.push(c.i64()?);
        }
        entity_phys.insert((0u8, tag), phys);
    }
    // Curves / surfaces / volumes: tag, 6-coord bbox, numPhysical, physical
    // tags, numBounding, bounding entity tags.
    for (dim, count) in [(1u8, nc), (2u8, ns), (3u8, nv)] {
        for _ in 0..count {
            let tag = c.i64()?;
            for _ in 0..6 {
                c.f64()?;
            }
            let k = c.usize()?;
            let mut phys = Vec::with_capacity(k);
            for _ in 0..k {
                phys.push(c.i64()?);
            }
            let nb = c.usize()?;
            for _ in 0..nb {
                c.i64()?;
            }
            entity_phys.insert((dim, tag), phys);
        }
    }
    Ok(entity_phys)
}

fn parse_v4(
    map: &HashMap<String, Vec<&str>>,
    names: &HashMap<(u8, i64), String>,
) -> Result<Parsed> {
    let entity_phys = parse_entities_v4(map)?;

    // $Nodes: blocks of (entityDim, entityTag, parametric, count), then the
    // node tags, then the coordinate triples.
    let nodes_sec = map
        .get("Nodes")
        .ok_or_else(|| err("gmsh: missing $Nodes"))?;
    let mut c = Cursor::new(nodes_sec);
    let nblocks = c.usize()?;
    let total = c.usize()?;
    let _min = c.u64()?;
    let _max = c.u64()?;
    let mut coords = HashMap::with_capacity(total);
    for _ in 0..nblocks {
        let _edim = c.i64()?;
        let _etag = c.i64()?;
        let parametric = c.usize()?;
        let cnt = c.usize()?;
        if parametric != 0 {
            return Err(err("gmsh: parametric nodes are not supported"));
        }
        let mut tags = Vec::with_capacity(cnt);
        for _ in 0..cnt {
            tags.push(c.u64()?);
        }
        for &tag in &tags {
            let xyz = [c.f64()?, c.f64()?, c.f64()?];
            coords.insert(tag, xyz);
        }
    }

    // $Elements: blocks of (entityDim, entityTag, elementType, count), then
    // one line `elemTag node…` per element.
    let elems_sec = map
        .get("Elements")
        .ok_or_else(|| err("gmsh: missing $Elements"))?;
    let mut c = Cursor::new(elems_sec);
    let nblocks = c.usize()?;
    let total = c.usize()?;
    let _min = c.u64()?;
    let _max = c.u64()?;
    let mut elements = Vec::with_capacity(total);
    for _ in 0..nblocks {
        let edim = c.i64()?;
        let etag = c.i64()?;
        let et = element_type_from_gmsh(c.u64()? as u32)?;
        let cnt = c.usize()?;
        let dim = edim as u8;
        let phys = entity_phys.get(&(dim, etag)).cloned().unwrap_or_default();
        let groups = groups_of(dim, &phys, names);
        let npc = et.nodes_per_cell();
        for _ in 0..cnt {
            let _id = c.u64()?;
            let mut nodes = Vec::with_capacity(npc);
            for _ in 0..npc {
                nodes.push(c.u64()?);
            }
            elements.push(ParsedElement {
                element_type: et,
                groups: groups.clone(),
                nodes,
            });
        }
    }
    Ok(Parsed { coords, elements })
}

// ─── Top-level parse + build ─────────────────────────────────────────────────

/// Parse the full text of a gmsh `.msh` file into the pure [`Parsed`] form.
fn parse_gmsh_text(text: &str) -> Result<Parsed> {
    let map = split_sections(text);
    let fmt = map
        .get("MeshFormat")
        .ok_or_else(|| err("gmsh: missing $MeshFormat (is this a .msh file?)"))?;
    let mut c = Cursor::new(fmt);
    let version = c.next_tok()?.to_string();
    let filetype = c.u64()?;
    if filetype != 0 {
        return Err(err(
            "gmsh: binary .msh files are not supported (ASCII only)",
        ));
    }
    let names = parse_physical_names(&map)?;
    match version.split('.').next() {
        Some("2") => parse_v2(&map, &names),
        Some("4") => parse_v4(&map, &names),
        _ => Err(err(format!(
            "gmsh: unsupported MSH version {version:?} (supported: 2.2, 4.1)"
        ))),
    }
}

/// Build the per-group meshes from the parsed data into the **caller's**
/// `coords`. The coordinate dimension is the one already carried by
/// `coords`: gmsh always stores three coordinates per node, of which the
/// first `coords.dim()` are kept (so a 2-D `Coords` flattens onto `xy`).
/// Groups come out in order of first appearance; within a group, submeshes
/// are ordered by the first cell of each element type.
fn build_groups(parsed: &Parsed, coords: Handle<Coords>) -> Result<Vec<(String, Mesh)>> {
    let dim = read(&coords)?.dim() as usize;

    // Materialize each referenced node once, keeping the `Node` alive in the
    // map so its refcount survives until every submesh has taken its own.
    let mut node_map: HashMap<u64, Node> = HashMap::new();

    // group name -> (element-type order, type -> submesh under construction).
    let mut order: Vec<String> = Vec::new();
    type Group = (Vec<ElementType>, HashMap<ElementType, SubMesh>);
    let mut groups: HashMap<String, Group> = HashMap::new();

    for el in &parsed.elements {
        // Resolve (creating as needed) the pyrucast node id of every tag.
        let mut ids: Vec<NodeId> = Vec::with_capacity(el.nodes.len());
        for &tag in &el.nodes {
            let id = match node_map.get(&tag) {
                Some(node) => node.id(),
                None => {
                    let xyz = parsed.coords.get(&tag).ok_or_else(|| {
                        err(format!("gmsh: element references unknown node {tag}"))
                    })?;
                    let node = Node::create_in(coords.clone(), &xyz[..dim])?;
                    let id = node.id();
                    node_map.insert(tag, node);
                    id
                }
            };
            ids.push(id);
        }

        for g in &el.groups {
            let entry = groups.entry(g.clone()).or_insert_with(|| {
                order.push(g.clone());
                (Vec::new(), HashMap::new())
            });
            if !entry.1.contains_key(&el.element_type) {
                entry.0.push(el.element_type);
                entry.1.insert(
                    el.element_type,
                    SubMesh::new(coords.clone(), el.element_type),
                );
            }
            entry.1.get_mut(&el.element_type).unwrap().add_cell(&ids)?;
        }
    }

    let mut out = Vec::with_capacity(order.len());
    for name in order {
        let (types, mut by_type) = groups.remove(&name).unwrap();
        let mut mesh = Mesh::empty();
        for et in types {
            mesh.add_sub(insert(by_type.remove(&et).unwrap()))?;
        }
        out.push((name, mesh));
    }
    Ok(out)
}

/// Read a gmsh `.msh` file (ASCII MSH 2.2 or 4.1) into one [`Mesh`] per
/// physical group, adding the nodes to the **caller's** `coords`. See the
/// module docs for the format and grouping rules.
///
/// The coordinate dimension is the one of `coords`: the first
/// `coords.dim()` of gmsh's three coordinates are kept. The nodes land in
/// `coords` (which may already hold geometry — the import is merged in), so
/// the caller keeps the handle it needs to pose boundary conditions etc.
pub fn read_gmsh(coords: Handle<Coords>, path: &Path) -> Result<Vec<(String, Mesh)>> {
    let text = std::fs::read_to_string(path)?;
    read_gmsh_str(coords, &text)
}

/// Like [`read_gmsh`] but parsing the file contents already held in memory.
pub fn read_gmsh_str(coords: Handle<Coords>, text: &str) -> Result<Vec<(String, Mesh)>> {
    let parsed = parse_gmsh_text(text)?;
    build_groups(&parsed, coords)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fresh `Coords` of the given dimension, to read into.
    fn coords(dim: u8) -> Handle<Coords> {
        insert(Coords::new(dim).unwrap())
    }

    // A unit square split into two TRI3, with a named surface "plate" and a
    // named bottom edge "bottom" (one SEG2). MSH 2.2 ASCII.
    const SQUARE_V2: &str = "\
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
1 1 \"bottom\"
2 2 \"plate\"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
3
1 1 2 1 1 1 2
2 2 2 2 2 1 2 3
3 2 2 2 2 1 3 4
$EndElements
";

    #[test]
    fn v2_groups_and_types() {
        let groups = read_gmsh_str(coords(2), SQUARE_V2).unwrap();
        let names: Vec<&str> = groups.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["bottom", "plate"]);

        let (_, bottom) = &groups[0];
        assert_eq!(bottom.element_types().unwrap(), vec![ElementType::SEG2]);
        assert_eq!(bottom.cell_count().unwrap(), 1);

        let (_, plate) = &groups[1];
        assert_eq!(plate.element_types().unwrap(), vec![ElementType::TRI3]);
        assert_eq!(plate.cell_count().unwrap(), 2);
    }

    #[test]
    fn reads_into_the_given_coords_shared_by_all_groups() {
        let c = coords(2);
        let groups = read_gmsh_str(c.clone(), SQUARE_V2).unwrap();
        // Nodes landed in the caller's Coords (the square's 4 corners).
        assert_eq!(read(&c).unwrap().node_count(), 4);
        // Every group hangs off that very same Coords slot.
        for (_, mesh) in &groups {
            let mc = mesh.coords().unwrap();
            assert_eq!(mc.index(), c.index());
            assert_eq!(mc.generation(), c.generation());
        }
    }

    // The same square in MSH 4.1: surface entity 1 → physical 2 ("plate"),
    // curve entity 1 → physical 1 ("bottom").
    const SQUARE_V4: &str = "\
$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
2
1 1 \"bottom\"
2 2 \"plate\"
$EndPhysicalNames
$Entities
0 1 1 0
1 0 0 0 1 0 0 1 1 0
1 0 0 0 1 1 0 1 2 0
$EndEntities
$Nodes
2 4 1 4
1 1 0 2
1
2
0 0 0
1 0 0
2 1 0 2
3
4
1 1 0
0 1 0
$EndNodes
$Elements
2 3 1 3
1 1 1 1
1 1 2
2 1 2 2
2 1 2 3
3 1 3 4
$EndElements
";

    #[test]
    fn v4_groups_and_types() {
        let groups = read_gmsh_str(coords(2), SQUARE_V4).unwrap();
        let names: Vec<&str> = groups.iter().map(|(n, _)| n.as_str()).collect();
        // "bottom" (curve) appears first in the elements block.
        assert_eq!(names, vec!["bottom", "plate"]);

        let (_, plate) = groups.iter().find(|(n, _)| n == "plate").unwrap();
        assert_eq!(plate.element_types().unwrap(), vec![ElementType::TRI3]);
        assert_eq!(plate.cell_count().unwrap(), 2);

        let (_, bottom) = groups.iter().find(|(n, _)| n == "bottom").unwrap();
        assert_eq!(bottom.element_types().unwrap(), vec![ElementType::SEG2]);
    }

    #[test]
    fn coords_dimension_decides_kept_coordinates() {
        let mesh = "\
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
2
1 0 0 5
2 1 0 7
$EndNodes
$Elements
1
1 1 2 0 1 1 2
$EndElements
";
        // A 3-D Coords keeps z.
        let g3 = read_gmsh_str(coords(3), mesh).unwrap();
        assert_eq!(
            g3[0].1.node(0, 0, 0).unwrap().coord().unwrap(),
            vec![0.0, 0.0, 5.0]
        );
        // A 2-D Coords keeps only x, y — z is dropped.
        let g2 = read_gmsh_str(coords(2), mesh).unwrap();
        assert_eq!(
            g2[0].1.node(0, 0, 0).unwrap().coord().unwrap(),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn ungrouped_elements_bucket() {
        let mesh = "\
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
2
1 0 0 0
2 1 0 0
$EndNodes
$Elements
1
1 1 2 0 1 1 2
$EndElements
";
        let groups = read_gmsh_str(coords(2), mesh).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, "<ungrouped>");
    }

    #[test]
    fn unsupported_element_type_errors() {
        let mesh = "\
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
3
1 0 0 0
2 1 0 0
3 2 0 0
$EndNodes
$Elements
1
1 8 2 0 1 1 2 3
$EndElements
";
        // gmsh type 8 = 3-node second-order line, not supported.
        assert!(read_gmsh_str(coords(2), mesh).is_err());
    }

    #[test]
    fn binary_is_rejected() {
        let mesh = "\
$MeshFormat
2.2 1 8
$EndMeshFormat
";
        let e = read_gmsh_str(coords(2), mesh).unwrap_err();
        assert!(matches!(e, PyrucastError::Message(m) if m.contains("binary")));
    }
}
