//! Read a gmsh `.msh` mesh into pyrucast meshes.
//!
//! Supported on-disk formats: **MSH 2.2** and **MSH 4.1**, in both their
//! **ASCII** (`$MeshFormat … 0 …`) and **binary** (`… 1 …`) variants. The
//! binary endianness is taken from the one-int marker gmsh writes in
//! `$MeshFormat`, so little- and big-endian files both read. The element
//! types pyrucast knows are mapped from their gmsh codes:
//!
//! | gmsh code | [`ElementType`] |
//! |---|---|
//! | `1`  | `SEG2` |
//! | `2`  | `TRI3` |
//! | `3`  | `QUA4` |
//! | `4`  | `TET4` |
//! | `5`  | `HEX8` |
//! | `6`  | `PENTA6` |
//! | `15` | `POI1` |
//! | `8`  | `SEG3` |
//! | `9`  | `TRI6` |
//! | `16` | `QUA8` |
//! | `10` | `QUA9` |
//! | `11` | `TET10` |
//! | `17` | `HEX20` |
//! | `18` | `PENTA15` |
//! | `12` | `HEX27` |
//!
//! Any other gmsh element type (order 3+, …) is an error. For most
//! types the local node ordering already matches pyrucast's reference frame
//! (see [`crate::atoms::ElementType`]) and the connectivity is
//! copied verbatim; the quadratic **volumes** (`TET10`, `HEX20`, `PENTA15`,
//! `HEX27`) have their mid-edge / face nodes **reordered** to pyrucast's
//! (VTK) order — see `gmsh_node_permutation`.
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
use crate::atoms::{ElementType, Node, NodeId};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::store::Handle;
use std::collections::HashMap;
use std::path::Path;

/// Synthetic group name for elements without a physical group.
const UNGROUPED: &str = "<ungrouped>";

fn err(msg: impl Into<String>) -> PyrucastError {
    PyrucastError::Message(msg.into())
}

/// Map a gmsh element-type code to a pyrucast [`ElementType`].
fn element_type_from_gmsh(code: u32) -> Result<ElementType> {
    ElementType::ALL
        .iter()
        .copied()
        .find(|et| et.as_kind().gmsh_code() == code)
        .ok_or_else(|| {
            let known: Vec<String> = ElementType::ALL
                .iter()
                .map(|et| format!("{}={et}", et.as_kind().gmsh_code()))
                .collect();
            err(format!(
                "gmsh: unsupported element type {code} (supported: {})",
                known.join(", ")
            ))
        })
}

/// Permutation mapping a gmsh element's node order to pyrucast's (VTK) order:
/// `pyrucast[i] = gmsh[perm[i]]`. Returns `None` when the two orders already
/// coincide (all linear types, plus `SEG3`/`TRI6`/`QUA8`/`QUA9`).
///
/// gmsh numbers the mid-edge nodes of `TET10`, `HEX20`, `PENTA15` and `HEX27`
/// (and the face nodes of `HEX27`) in a different order than VTK; these tables
/// realign them (same convention as meshio).
fn gmsh_node_permutation(et: ElementType) -> Option<&'static [usize]> {
    et.as_kind().gmsh_permutation()
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
        parse_physical_name_line(line, &mut names)?;
    }
    Ok(names)
}

/// Parse one `$PhysicalNames` entry — `dim tag "name"` — into `names`.
fn parse_physical_name_line(line: &str, names: &mut HashMap<(u8, i64), String>) -> Result<()> {
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
    Ok(())
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
    // file-type is handled by the binary/ASCII dispatch in `parse_gmsh`;
    // here we only need the version to pick the v2/v4 section layout.
    let mut c = Cursor::new(fmt);
    let version = c.next_tok()?.to_string();
    let names = parse_physical_names(&map)?;
    match version.split('.').next() {
        Some("2") => parse_v2(&map, &names),
        Some("4") => parse_v4(&map, &names),
        _ => Err(err(format!(
            "gmsh: unsupported MSH version {version:?} (supported: 2.2, 4.1)"
        ))),
    }
}

// ─── Binary MSH ──────────────────────────────────────────────────────────────

/// Byte cursor over a binary `.msh` file. Reads ASCII lines (section
/// markers and the count lines that gmsh keeps textual) and fixed-width
/// little-/big-endian scalars (the binary payloads).
struct ByteReader<'a> {
    buf: &'a [u8],
    pos: usize,
    le: bool,
}

impl<'a> ByteReader<'a> {
    /// Next `\n`-terminated line, trimmed of surrounding ASCII whitespace
    /// (incl. `\r`). `None` at end of buffer.
    fn try_line(&mut self) -> Option<&'a [u8]> {
        if self.pos >= self.buf.len() {
            return None;
        }
        let rest = &self.buf[self.pos..];
        let line = match rest.iter().position(|&b| b == b'\n') {
            Some(i) => {
                self.pos += i + 1;
                &rest[..i]
            }
            None => {
                self.pos = self.buf.len();
                rest
            }
        };
        Some(line.trim_ascii())
    }

    fn line(&mut self) -> Result<&'a [u8]> {
        self.try_line()
            .ok_or_else(|| err("gmsh: unexpected end of file"))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| err("gmsh: truncated binary data"))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn i32(&mut self) -> Result<i32> {
        let b: [u8; 4] = self.take(4)?.try_into().unwrap();
        Ok(if self.le {
            i32::from_le_bytes(b)
        } else {
            i32::from_be_bytes(b)
        })
    }

    /// A gmsh `size_t` (8 bytes).
    fn size_t(&mut self) -> Result<u64> {
        let b: [u8; 8] = self.take(8)?.try_into().unwrap();
        Ok(if self.le {
            u64::from_le_bytes(b)
        } else {
            u64::from_be_bytes(b)
        })
    }

    fn f64(&mut self) -> Result<f64> {
        let b: [u8; 8] = self.take(8)?.try_into().unwrap();
        Ok(if self.le {
            f64::from_le_bytes(b)
        } else {
            f64::from_be_bytes(b)
        })
    }

    /// Consume lines until one equals `marker` (the section's `$End…`).
    fn skip_to(&mut self, marker: &[u8]) -> Result<()> {
        loop {
            if self.line()? == marker {
                return Ok(());
            }
        }
    }
}

/// Parse a non-empty ASCII line as an integer count.
fn ascii_count(line: &[u8]) -> Result<usize> {
    std::str::from_utf8(line)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .ok_or_else(|| err("gmsh: expected an integer count"))
}

/// Read the version major and file-type from `$MeshFormat` without decoding
/// the rest — enough to choose the ASCII or binary path.
fn peek_format(buf: &[u8]) -> Result<(u32, u64)> {
    let mut r = ByteReader {
        buf,
        pos: 0,
        le: true,
    };
    loop {
        if r.line()? == b"$MeshFormat" {
            break;
        }
    }
    let line = std::str::from_utf8(r.line()?).map_err(|_| err("gmsh: bad $MeshFormat line"))?;
    let mut it = line.split_whitespace();
    let version = it.next().ok_or_else(|| err("gmsh: empty $MeshFormat"))?;
    let filetype: u64 = it
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| err("gmsh: $MeshFormat without a file-type"))?;
    let major: u32 = version
        .split('.')
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| err(format!("gmsh: unparsable MSH version {version:?}")))?;
    Ok((major, filetype))
}

/// Parse a binary `.msh` (`file-type 1`), MSH major version 2 or 4.
fn parse_binary(buf: &[u8], major: u32) -> Result<Parsed> {
    let mut r = ByteReader {
        buf,
        pos: 0,
        le: true,
    };
    // Re-reach $MeshFormat, skip the version line, then read the endianness
    // marker: an int written as `1` in the file's own byte order.
    loop {
        if r.line()? == b"$MeshFormat" {
            break;
        }
    }
    r.line()?; // version / file-type / data-size
    let marker = r.take(4)?.try_into().unwrap();
    r.le = i32::from_le_bytes(marker) == 1;
    if !r.le && i32::from_be_bytes(marker) != 1 {
        return Err(err("gmsh: unrecognized binary endianness marker"));
    }
    r.skip_to(b"$EndMeshFormat")?;

    let mut names: HashMap<(u8, i64), String> = HashMap::new();
    let mut entity_phys: HashMap<(u8, i64), Vec<i64>> = HashMap::new();
    let mut parsed = Parsed {
        coords: HashMap::new(),
        elements: Vec::new(),
    };

    while let Some(line) = r.try_line() {
        if line.is_empty() {
            continue;
        }
        match line {
            b"$PhysicalNames" => {
                let count = ascii_count(r.line()?)?;
                for _ in 0..count {
                    let l =
                        std::str::from_utf8(r.line()?).map_err(|_| err("gmsh: bad name line"))?;
                    parse_physical_name_line(l, &mut names)?;
                }
                r.skip_to(b"$EndPhysicalNames")?;
            }
            b"$Entities" if major == 4 => {
                entity_phys = parse_entities_binary(&mut r)?;
            }
            b"$Nodes" if major == 2 => parse_nodes_v2_binary(&mut r, &mut parsed.coords)?,
            b"$Nodes" => parse_nodes_v4_binary(&mut r, &mut parsed.coords)?,
            b"$Elements" if major == 2 => {
                parse_elements_v2_binary(&mut r, &names, &mut parsed.elements)?
            }
            b"$Elements" => {
                parse_elements_v4_binary(&mut r, &entity_phys, &names, &mut parsed.elements)?
            }
            other if other.starts_with(b"$") && !other.starts_with(b"$End") => {
                // Unknown section: skip to its matching $End… marker.
                let mut end = b"$End".to_vec();
                end.extend_from_slice(&other[1..]);
                r.skip_to(&end)?;
            }
            _ => {}
        }
    }
    Ok(parsed)
}

fn parse_entities_binary(r: &mut ByteReader) -> Result<HashMap<(u8, i64), Vec<i64>>> {
    let mut entity_phys = HashMap::new();
    let np = r.size_t()?;
    let nc = r.size_t()?;
    let ns = r.size_t()?;
    let nv = r.size_t()?;
    for _ in 0..np {
        let tag = r.i32()? as i64;
        for _ in 0..3 {
            r.f64()?;
        }
        let k = r.size_t()?;
        let phys = (0..k)
            .map(|_| r.i32().map(|t| t as i64))
            .collect::<Result<_>>()?;
        entity_phys.insert((0u8, tag), phys);
    }
    for (dim, count) in [(1u8, nc), (2u8, ns), (3u8, nv)] {
        for _ in 0..count {
            let tag = r.i32()? as i64;
            for _ in 0..6 {
                r.f64()?;
            }
            let k = r.size_t()?;
            let phys = (0..k)
                .map(|_| r.i32().map(|t| t as i64))
                .collect::<Result<_>>()?;
            let nb = r.size_t()?;
            for _ in 0..nb {
                r.i32()?;
            }
            entity_phys.insert((dim, tag), phys);
        }
    }
    r.skip_to(b"$EndEntities")?;
    Ok(entity_phys)
}

fn parse_nodes_v2_binary(r: &mut ByteReader, coords: &mut HashMap<u64, [f64; 3]>) -> Result<()> {
    let n = ascii_count(r.line()?)?;
    for _ in 0..n {
        let tag = r.i32()? as u64;
        let xyz = [r.f64()?, r.f64()?, r.f64()?];
        coords.insert(tag, xyz);
    }
    r.skip_to(b"$EndNodes")
}

fn parse_elements_v2_binary(
    r: &mut ByteReader,
    names: &HashMap<(u8, i64), String>,
    elements: &mut Vec<ParsedElement>,
) -> Result<()> {
    let total = ascii_count(r.line()?)?;
    let mut done = 0;
    while done < total {
        // Block header: element type, count in block, number of tags.
        let et = element_type_from_gmsh(r.i32()? as u32)?;
        let num_in_block = r.i32()? as usize;
        let num_tags = r.i32()? as usize;
        let npc = et.nodes_per_cell();
        for _ in 0..num_in_block {
            let _id = r.i32()?;
            let mut phys: Vec<i64> = Vec::new();
            for k in 0..num_tags {
                let t = r.i32()?;
                if k == 0 && t != 0 {
                    phys.push(t as i64);
                }
            }
            let nodes = (0..npc)
                .map(|_| r.i32().map(|v| v as u64))
                .collect::<Result<_>>()?;
            let groups = groups_of(et.topological_dim() as u8, &phys, names);
            elements.push(ParsedElement {
                element_type: et,
                groups,
                nodes,
            });
        }
        done += num_in_block;
    }
    r.skip_to(b"$EndElements")
}

fn parse_nodes_v4_binary(r: &mut ByteReader, coords: &mut HashMap<u64, [f64; 3]>) -> Result<()> {
    let nblocks = r.size_t()?;
    let _total = r.size_t()?;
    let _min = r.size_t()?;
    let _max = r.size_t()?;
    for _ in 0..nblocks {
        let _edim = r.i32()?;
        let _etag = r.i32()?;
        let parametric = r.i32()?;
        let cnt = r.size_t()? as usize;
        if parametric != 0 {
            return Err(err("gmsh: parametric nodes are not supported"));
        }
        let tags: Vec<u64> = (0..cnt).map(|_| r.size_t()).collect::<Result<_>>()?;
        for &tag in &tags {
            let xyz = [r.f64()?, r.f64()?, r.f64()?];
            coords.insert(tag, xyz);
        }
    }
    r.skip_to(b"$EndNodes")
}

fn parse_elements_v4_binary(
    r: &mut ByteReader,
    entity_phys: &HashMap<(u8, i64), Vec<i64>>,
    names: &HashMap<(u8, i64), String>,
    elements: &mut Vec<ParsedElement>,
) -> Result<()> {
    let nblocks = r.size_t()?;
    let _total = r.size_t()?;
    let _min = r.size_t()?;
    let _max = r.size_t()?;
    for _ in 0..nblocks {
        let edim = r.i32()?;
        let etag = r.i32()? as i64;
        let et = element_type_from_gmsh(r.i32()? as u32)?;
        let cnt = r.size_t()? as usize;
        let dim = edim as u8;
        let phys = entity_phys.get(&(dim, etag)).cloned().unwrap_or_default();
        let groups = groups_of(dim, &phys, names);
        let npc = et.nodes_per_cell();
        for _ in 0..cnt {
            let _tag = r.size_t()?;
            let nodes = (0..npc).map(|_| r.size_t()).collect::<Result<_>>()?;
            elements.push(ParsedElement {
                element_type: et,
                groups: groups.clone(),
                nodes,
            });
        }
    }
    r.skip_to(b"$EndElements")
}

/// Parse a gmsh `.msh` file (ASCII or binary) into the pure [`Parsed`] form.
fn parse_gmsh(buf: &[u8]) -> Result<Parsed> {
    let (major, filetype) = peek_format(buf)?;
    match filetype {
        0 => {
            let text = std::str::from_utf8(buf)
                .map_err(|_| err("gmsh: file declared ASCII but is not valid UTF-8"))?;
            parse_gmsh_text(text)
        }
        1 => parse_binary(buf, major),
        other => Err(err(format!(
            "gmsh: unsupported file-type {other} (0 = ASCII, 1 = binary)"
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
    let dim = coords.read().dim() as usize;

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

        // Realign gmsh's mid-edge node order to pyrucast's where they differ.
        if let Some(perm) = gmsh_node_permutation(el.element_type) {
            if ids.len() == perm.len() {
                ids = perm.iter().map(|&p| ids[p]).collect();
            }
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
            mesh.add_sub(Handle::new(by_type.remove(&et).unwrap()))?;
        }
        out.push((name, mesh));
    }
    Ok(out)
}

/// Read a gmsh `.msh` file (ASCII or binary, MSH 2.2 or 4.1) into one
/// [`Mesh`] per physical group, adding the nodes to the **caller's**
/// `coords`. See the module docs for the format and grouping rules.
///
/// The coordinate dimension is the one of `coords`: the first
/// `coords.dim()` of gmsh's three coordinates are kept. The nodes land in
/// `coords` (which may already hold geometry — the import is merged in), so
/// the caller keeps the handle it needs to pose boundary conditions etc.
pub fn read_gmsh(coords: Handle<Coords>, path: &Path) -> Result<Vec<(String, Mesh)>> {
    let bytes = std::fs::read(path)?;
    read_gmsh_bytes(coords, &bytes)
}

/// Like [`read_gmsh`] but parsing **ASCII** file contents already held in a
/// string. For binary content use [`read_gmsh_bytes`].
pub fn read_gmsh_str(coords: Handle<Coords>, text: &str) -> Result<Vec<(String, Mesh)>> {
    read_gmsh_bytes(coords, text.as_bytes())
}

/// Like [`read_gmsh`] but parsing the raw file bytes already held in memory
/// (ASCII or binary).
pub fn read_gmsh_bytes(coords: Handle<Coords>, bytes: &[u8]) -> Result<Vec<(String, Mesh)>> {
    let parsed = parse_gmsh(bytes)?;
    build_groups(&parsed, coords)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fresh `Coords` of the given dimension, to read into.
    fn coords(dim: u8) -> Handle<Coords> {
        Handle::new(Coords::new(dim).unwrap())
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
        assert_eq!(c.read().node_count(), 4);
        // Every group hangs off that very same Coords slot.
        for (_, mesh) in &groups {
            let mc = mesh.coords().unwrap();
            assert!(mc.same_object(&c));
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
            g3[0].1.node(0, 0, 0).unwrap().position().unwrap(),
            vec![0.0, 0.0, 5.0]
        );
        // A 2-D Coords keeps only x, y — z is dropped.
        let g2 = read_gmsh_str(coords(2), mesh).unwrap();
        assert_eq!(
            g2[0].1.node(0, 0, 0).unwrap().position().unwrap(),
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
1 21 2 0 1 1 2 3
$EndElements
";
        // gmsh type 21 = 10-node third-order triangle, which pyrucast has no
        // element for.
        let err = read_gmsh_str(coords(2), mesh).unwrap_err();
        assert!(
            format!("{err}").contains("unsupported element type 21"),
            "{err}"
        );
    }

    #[test]
    fn reads_a_pyramid() {
        // gmsh type 7 = 5-node pyramid: square base then apex, the same order
        // pyrucast uses.
        let mesh = "\
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
5
1 -1 -1 0
2 1 -1 0
3 1 1 0
4 -1 1 0
5 0 0 1
$EndNodes
$Elements
1
1 7 2 0 1 1 2 3 4 5
$EndElements
";
        let groups = read_gmsh_str(coords(3), mesh).unwrap();
        assert_eq!(groups.len(), 1);
        let m = &groups[0].1;
        assert_eq!(m.element_types().unwrap(), vec![ElementType::PYRA5]);
        assert_eq!(m.cell_count().unwrap(), 1);
        assert_eq!(
            m.node(0, 0, 4).unwrap().position().unwrap(),
            vec![0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn reads_quadratic_tet10_with_node_permutation() {
        // A single TET10 (gmsh type 11) on the unit tetrahedron, its 6
        // mid-edge nodes at the exact edge midpoints, listed in **gmsh**
        // node order (edges (0,1),(1,2),(2,0),(0,3),(2,3),(1,3)). After the
        // gmsh→pyrucast permutation, local node 8 must be the (1,3) midpoint
        // and node 9 the (2,3) midpoint.
        let mesh = "\
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
10
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
5 0.5 0 0
6 0.5 0.5 0
7 0 0.5 0
8 0 0 0.5
9 0 0.5 0.5
10 0.5 0 0.5
$EndNodes
$Elements
1
1 11 2 0 1 1 2 3 4 5 6 7 8 9 10
$EndElements
";
        let g = read_gmsh_str(coords(3), mesh).unwrap();
        let (_, m) = &g[0];
        assert_eq!(m.element_types().unwrap(), vec![ElementType::TET10]);
        assert_eq!(m.cell_count().unwrap(), 1);
        // Local node 8 = edge (1,3) midpoint, node 9 = edge (2,3) midpoint.
        assert_eq!(
            m.node(0, 0, 8).unwrap().position().unwrap(),
            vec![0.5, 0.0, 0.5]
        );
        assert_eq!(
            m.node(0, 0, 9).unwrap().position().unwrap(),
            vec![0.0, 0.5, 0.5]
        );
    }

    #[test]
    fn reads_quadratic_hex27_with_node_permutation() {
        // A single HEX27 (gmsh type 12) on the unit cube, every mid-edge /
        // face / center node at its exact geometric position, listed in
        // **gmsh** node order. After the gmsh→pyrucast permutation the face
        // centers (local 20..25) and body center (26) must land correctly.
        let mesh = "\
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
27
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
5 0 0 1
6 1 0 1
7 1 1 1
8 0 1 1
9 0.5 0 0
10 0 0.5 0
11 0 0 0.5
12 1 0.5 0
13 1 0 0.5
14 0.5 1 0
15 1 1 0.5
16 0 1 0.5
17 0.5 0 1
18 0 0.5 1
19 1 0.5 1
20 0.5 1 1
21 0.5 0.5 0
22 0.5 0 0.5
23 0 0.5 0.5
24 1 0.5 0.5
25 0.5 1 0.5
26 0.5 0.5 1
27 0.5 0.5 0.5
$EndNodes
$Elements
1
1 12 2 0 1 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
$EndElements
";
        let g = read_gmsh_str(coords(3), mesh).unwrap();
        let (_, m) = &g[0];
        assert_eq!(m.element_types().unwrap(), vec![ElementType::HEX27]);
        // Faces x-, x+, y-, y+, z-, z+ at local 20..25; body center at 26.
        assert_eq!(
            m.node(0, 0, 20).unwrap().position().unwrap(),
            vec![0.0, 0.5, 0.5]
        );
        assert_eq!(
            m.node(0, 0, 21).unwrap().position().unwrap(),
            vec![1.0, 0.5, 0.5]
        );
        assert_eq!(
            m.node(0, 0, 22).unwrap().position().unwrap(),
            vec![0.5, 0.0, 0.5]
        );
        assert_eq!(
            m.node(0, 0, 23).unwrap().position().unwrap(),
            vec![0.5, 1.0, 0.5]
        );
        assert_eq!(
            m.node(0, 0, 24).unwrap().position().unwrap(),
            vec![0.5, 0.5, 0.0]
        );
        assert_eq!(
            m.node(0, 0, 25).unwrap().position().unwrap(),
            vec![0.5, 0.5, 1.0]
        );
        assert_eq!(
            m.node(0, 0, 26).unwrap().position().unwrap(),
            vec![0.5, 0.5, 0.5]
        );
    }

    // ─── Binary round-trips ──────────────────────────────────────────────────
    //
    // The same unit square as SQUARE_V2 / SQUARE_V4, hand-encoded in
    // little-endian binary, must read identically to its ASCII twin.

    // Endianness-parametrized scalar encoders, so the same fixtures exercise
    // both the little- and big-endian read paths.
    fn i32e(b: &mut Vec<u8>, x: i32, le: bool) {
        b.extend_from_slice(&if le { x.to_le_bytes() } else { x.to_be_bytes() });
    }
    fn u64e(b: &mut Vec<u8>, x: u64, le: bool) {
        b.extend_from_slice(&if le { x.to_le_bytes() } else { x.to_be_bytes() });
    }
    fn f64e(b: &mut Vec<u8>, x: f64, le: bool) {
        b.extend_from_slice(&if le { x.to_le_bytes() } else { x.to_be_bytes() });
    }

    fn square_v2_binary(le: bool) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"$MeshFormat\n2.2 1 8\n");
        i32e(&mut b, 1, le); // endianness marker
        b.extend_from_slice(b"\n$EndMeshFormat\n");
        b.extend_from_slice(
            b"$PhysicalNames\n2\n1 1 \"bottom\"\n2 2 \"plate\"\n$EndPhysicalNames\n",
        );
        b.extend_from_slice(b"$Nodes\n4\n");
        for (tag, x, y) in [(1, 0.0, 0.0), (2, 1.0, 0.0), (3, 1.0, 1.0), (4, 0.0, 1.0)] {
            i32e(&mut b, tag, le);
            f64e(&mut b, x, le);
            f64e(&mut b, y, le);
            f64e(&mut b, 0.0, le);
        }
        b.extend_from_slice(b"\n$EndNodes\n");
        b.extend_from_slice(b"$Elements\n3\n");
        // SEG2 block (type 1, 1 element, 2 tags): elem 1, phys 1, geom 1, nodes 1-2.
        i32e(&mut b, 1, le);
        i32e(&mut b, 1, le);
        i32e(&mut b, 2, le);
        for v in [1, 1, 1, 1, 2] {
            i32e(&mut b, v, le);
        }
        // TRI3 block (type 2, 2 elements, 2 tags).
        i32e(&mut b, 2, le);
        i32e(&mut b, 2, le);
        i32e(&mut b, 2, le);
        for v in [2, 2, 2, 1, 2, 3] {
            i32e(&mut b, v, le);
        }
        for v in [3, 2, 2, 1, 3, 4] {
            i32e(&mut b, v, le);
        }
        b.extend_from_slice(b"\n$EndElements\n");
        b
    }

    fn square_v4_binary(le: bool) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"$MeshFormat\n4.1 1 8\n");
        i32e(&mut b, 1, le);
        b.extend_from_slice(b"\n$EndMeshFormat\n");
        b.extend_from_slice(
            b"$PhysicalNames\n2\n1 1 \"bottom\"\n2 2 \"plate\"\n$EndPhysicalNames\n",
        );
        // Entities: 0 points, 1 curve, 1 surface, 0 volumes.
        b.extend_from_slice(b"$Entities\n");
        for n in [0u64, 1, 1, 0] {
            u64e(&mut b, n, le);
        }
        // curve tag 1: bbox(6 f64), 1 physical (1), 0 bounding.
        i32e(&mut b, 1, le);
        for _ in 0..6 {
            f64e(&mut b, 0.0, le);
        }
        u64e(&mut b, 1, le);
        i32e(&mut b, 1, le);
        u64e(&mut b, 0, le);
        // surface tag 1: bbox(6 f64), 1 physical (2), 0 bounding.
        i32e(&mut b, 1, le);
        for _ in 0..6 {
            f64e(&mut b, 0.0, le);
        }
        u64e(&mut b, 1, le);
        i32e(&mut b, 2, le);
        u64e(&mut b, 0, le);
        b.extend_from_slice(b"\n$EndEntities\n");
        // Nodes: 2 blocks, 4 nodes total.
        b.extend_from_slice(b"$Nodes\n");
        for n in [2u64, 4, 1, 4] {
            u64e(&mut b, n, le);
        }
        // block curve(dim1,tag1), parametric 0, 2 nodes (1,2).
        i32e(&mut b, 1, le);
        i32e(&mut b, 1, le);
        i32e(&mut b, 0, le);
        u64e(&mut b, 2, le);
        u64e(&mut b, 1, le);
        u64e(&mut b, 2, le);
        for (x, y) in [(0.0, 0.0), (1.0, 0.0)] {
            f64e(&mut b, x, le);
            f64e(&mut b, y, le);
            f64e(&mut b, 0.0, le);
        }
        // block surface(dim2,tag1), parametric 0, 2 nodes (3,4).
        i32e(&mut b, 2, le);
        i32e(&mut b, 1, le);
        i32e(&mut b, 0, le);
        u64e(&mut b, 2, le);
        u64e(&mut b, 3, le);
        u64e(&mut b, 4, le);
        for (x, y) in [(1.0, 1.0), (0.0, 1.0)] {
            f64e(&mut b, x, le);
            f64e(&mut b, y, le);
            f64e(&mut b, 0.0, le);
        }
        b.extend_from_slice(b"\n$EndNodes\n");
        // Elements: 2 blocks, 3 elements total.
        b.extend_from_slice(b"$Elements\n");
        for n in [2u64, 3, 1, 3] {
            u64e(&mut b, n, le);
        }
        // SEG2 block on curve(dim1,tag1): 1 element, tag 1, nodes 1-2.
        i32e(&mut b, 1, le);
        i32e(&mut b, 1, le);
        i32e(&mut b, 1, le);
        u64e(&mut b, 1, le);
        u64e(&mut b, 1, le);
        u64e(&mut b, 1, le);
        u64e(&mut b, 2, le);
        // TRI3 block on surface(dim2,tag1): 2 elements.
        i32e(&mut b, 2, le);
        i32e(&mut b, 1, le);
        i32e(&mut b, 2, le);
        u64e(&mut b, 2, le);
        u64e(&mut b, 2, le);
        for v in [1u64, 2, 3] {
            u64e(&mut b, v, le);
        }
        u64e(&mut b, 3, le);
        for v in [1u64, 3, 4] {
            u64e(&mut b, v, le);
        }
        b.extend_from_slice(b"\n$EndElements\n");
        b
    }

    /// Shared assertions: the binary square reads to the same groups/types
    /// as its ASCII twin, with coordinates decoded correctly.
    fn assert_square(groups: &[(String, Mesh)]) {
        let names: Vec<&str> = groups.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["bottom", "plate"]);
        let (_, plate) = groups.iter().find(|(n, _)| n == "plate").unwrap();
        assert_eq!(plate.element_types().unwrap(), vec![ElementType::TRI3]);
        assert_eq!(plate.cell_count().unwrap(), 2);
        let (_, bottom) = groups.iter().find(|(n, _)| n == "bottom").unwrap();
        assert_eq!(bottom.element_types().unwrap(), vec![ElementType::SEG2]);
        assert_eq!(bottom.cell_count().unwrap(), 1);
        // node 3 sits at (1, 1) — checks the f64 payload decoded right.
        assert_eq!(
            plate.node(0, 0, 2).unwrap().position().unwrap(),
            vec![1.0, 1.0]
        );
    }

    #[test]
    fn binary_v2_matches_ascii() {
        assert_square(&read_gmsh_bytes(coords(2), &square_v2_binary(true)).unwrap());
    }

    #[test]
    fn binary_v4_matches_ascii() {
        assert_square(&read_gmsh_bytes(coords(2), &square_v4_binary(true)).unwrap());
    }

    #[test]
    fn binary_big_endian_round_trips() {
        // Same fixtures encoded big-endian; the marker drives detection.
        assert_square(&read_gmsh_bytes(coords(2), &square_v2_binary(false)).unwrap());
        assert_square(&read_gmsh_bytes(coords(2), &square_v4_binary(false)).unwrap());
    }

    #[test]
    fn binary_bad_endianness_marker_errors() {
        let mut bad = square_v2_binary(true);
        // Corrupt the endianness marker (the int right after "2.2 1 8\n").
        let at = bad.windows(8).position(|w| w == b"2.2 1 8\n").unwrap() + 8;
        bad[at] = 9; // neither 1 LE nor 1 BE
        let e = read_gmsh_bytes(coords(2), &bad).unwrap_err();
        assert!(matches!(e, PyrucastError::Message(m) if m.contains("endianness")));
    }
}
