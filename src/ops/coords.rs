//! Operators on the **coordinates** themselves: the only three that write
//! into them, and the one that computes in another configuration.
//!
//! The writers are the write side of
//! [`node_field::positions`](fn@crate::ops::node_field::positions), which
//! reads the geometry as a nodal field: [`set`](fn@set) puts absolute
//! positions back, [`displace`](fn@displace) adds an increment, and
//! [`displace_from`](fn@displace_from) lays one configuration at another
//! plus a displacement. All act on the [`Coords`] the
//! field hangs from, in place. [`in_config`](fn@in_config) runs a computation
//! with another configuration active, and makes the one before it active
//! again.

use crate::aggregate::Aggregate;
use crate::atoms::NodeId;
use crate::containers::field::{Field, SubField};
use crate::containers::node_field::{first_rows, NodeField, SubNodeField};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::handle::{Handle, ReadGuard};

/// Resolve the per-axis component names: one name per spatial axis, in axis
/// order. `None` falls back to `default[..dim]`. Errors if the count does
/// not match `dim` or if a name is absent from `field` (union of zones).
fn resolve_axis_components(
    field: &NodeField,
    components: Option<Vec<String>>,
    dim: usize,
    default: &[&str],
) -> Result<Vec<String>> {
    let comps = match components {
        Some(c) => c,
        None => default[..dim.min(default.len())]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };
    if comps.len() != dim {
        return Err(PyrucastError::Message(format!(
            "expected {dim} component name(s) (one per axis), got {}",
            comps.len()
        )));
    }
    let have = Field::components(field);
    for name in &comps {
        if !have.iter().any(|c| c == name) {
            return Err(PyrucastError::Message(format!(
                "component '{name}' not in field (has: [{}])",
                have.join(", ")
            )));
        }
    }
    Ok(comps)
}

/// Every distinct node of `field`, first-seen order, and its values of
/// `comps`, `comps.len()` per node — read through the aggregate (first zone
/// wins), so interface nodes appear exactly once. Errors if a node lacks one
/// of the components.
fn per_node_values(field: &NodeField, comps: &[String]) -> Result<(Vec<NodeId>, Vec<f64>)> {
    let (mut nodes, mut values) = (Vec::new(), Vec::new());
    // (node, component) the first zone holding the node lacks: some later zone
    // may hold it.
    let mut missing: Vec<(usize, usize)> = Vec::new();
    {
        let guards: Vec<ReadGuard<SubNodeField>> = field.iter().map(|h| h.read()).collect();
        let zones: Vec<&SubNodeField> = guards.iter().map(|g| &**g).collect();
        first_rows(&zones, |z, rows| {
            let (held, width) = (zones[z].values(), zones[z].component_count());
            let cols: Vec<Option<usize>> =
                comps.iter().map(|c| zones[z].component_index(c)).collect();
            // At most the zone's rows are new: one growth per zone.
            nodes.reserve(zones[z].node_count());
            values.reserve(zones[z].node_count() * comps.len());
            if let Some(cols) = cols.iter().copied().collect::<Option<Vec<usize>>>() {
                rows.for_each(|row, p| {
                    let r = &held[row * width..(row + 1) * width];
                    nodes.push(p);
                    values.extend(cols.iter().map(|&c| r[c]));
                });
                return;
            }
            rows.for_each(|row, p| {
                for (k, col) in cols.iter().enumerate() {
                    match col {
                        Some(c) => values.push(held[row * width + c]),
                        None => {
                            missing.push((nodes.len(), k));
                            values.push(0.0);
                        }
                    }
                }
                nodes.push(p);
            });
        });
    }
    if !missing.is_empty() {
        let view = field.view()?;
        for (i, k) in missing {
            values[i * comps.len() + k] = view.value(nodes[i], &comps[k])?;
        }
    }
    Ok((nodes, values))
}

/// **Set** node coordinates from `field` (absolute): for every distinct
/// node `n` of the field, `coord[a] = field.value(n, components[a])` on
/// the active coordinate set. `components` lists one field-component name
/// per spatial axis, in axis order; `None` → `["X", "Y", "Z"][..dim]`
/// (symmetric with [`crate::ops::node_field::positions`](fn@crate::ops::node_field::positions)). In-place on the field's
/// `Coords`.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{Band, ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{coords as ops_coords, element_field, field, measure, mesh, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// # let champ = |noms: Vec<String>| {
/// #     NodeField::from_submesh(&support.get(0).unwrap(), noms).unwrap()
/// # };
/// # let temp = champ(vec!["T".into()]);
/// # {
/// #     let mut z = temp.get(0).unwrap().write();
/// #     z.set_value(n[0].id(), "T", 10.0).unwrap();
/// #     z.set_value(n[1].id(), "T", 50.0).unwrap();
/// #     z.set_value(n[2].id(), "T", 90.0).unwrap();
/// # }
/// // Writes the positions **into the frame**: the inverse of
/// // `node_field::positions`. It is an assumed side effect.
/// // The default components are `X`, `Y`, `Z` — those that
/// // `positions` produit.
/// let cible = node_field::positions(&maillage, None)?;
/// cible.get(0)?.write().add_to_component("X", 1.0)?;
/// ops_coords::set(&cible, None)?;
/// assert_eq!(n[0].position()?, vec![1.0, 0.0]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn set(field: &NodeField, components: Option<Vec<String>>) -> Result<()> {
    let coords = field.coords()?;
    let dim = coords.read().dim() as usize;
    let comps = resolve_axis_components(field, components, dim, &["X", "Y", "Z"])?;
    let (nodes, targets) = per_node_values(field, &comps)?;
    let mut c = coords.write();
    for (&nid, coord) in nodes.iter().zip(targets.chunks_exact(dim)) {
        c.set_position(nid, coord)?;
    }
    Ok(())
}

/// **Displace** nodes by `field` (incremental): for every distinct node
/// `n`, `coord[a] += field.value(n, components[a])` on the active
/// coordinate set — an interface node shared by several zones is displaced
/// exactly once. `components` lists one displacement-component name per
/// spatial axis, in axis order; `None` → `["ux", "uy", "uz"][..dim]`.
/// In-place on the field's `Coords`.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{Band, ElementType, Node};
/// # use pyrucast::containers::element_field::ElementField;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::finite_element_space::FiniteElementSpace;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::{coords as ops_coords, element_field, field, measure, mesh, node_field};
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # let fes = FiniteElementSpace::lagrange1(&maillage).unwrap();
/// # let support = mesh::poi1_from_nodes(&n).unwrap();
/// # let champ = |noms: Vec<String>| {
/// #     NodeField::from_submesh(&support.get(0).unwrap(), noms).unwrap()
/// # };
/// # let temp = champ(vec!["T".into()]);
/// # {
/// #     let mut z = temp.get(0).unwrap().write();
/// #     z.set_value(n[0].id(), "T", 10.0).unwrap();
/// #     z.set_value(n[1].id(), "T", 50.0).unwrap();
/// #     z.set_value(n[2].id(), "T", 90.0).unwrap();
/// # }
/// // Adds a displacement to the positions instead of replacing them — the
/// // façon de passer en configuration déformée.
/// // The default components here are `ux`, `uy`, `uz`.
/// let u = champ(vec!["ux".into(), "uy".into()]);
/// u.get(0)?.write().add_to_component("ux", 0.5)?;
/// ops_coords::displace(&u, None)?;
/// assert_eq!(n[0].position()?, vec![0.5, 0.0]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn displace(field: &NodeField, components: Option<Vec<String>>) -> Result<()> {
    let coords = field.coords()?;
    let dim = coords.read().dim() as usize;
    let comps = resolve_axis_components(field, components, dim, &["ux", "uy", "uz"])?;
    let (nodes, increments) = per_node_values(field, &comps)?;
    let mut c = coords.write();
    let mut coord = vec![0.0; dim];
    for (&nid, inc) in nodes.iter().zip(increments.chunks_exact(dim)) {
        coord.copy_from_slice(c.position(nid)?);
        for (x, dv) in coord.iter_mut().zip(inc) {
            *x += dv;
        }
        c.set_position(nid, &coord)?;
    }
    Ok(())
}

/// Lay configuration `target` at configuration `reference` displaced by
/// `field` — `x = X + u`, in **absolute**: each node takes its `reference`
/// position plus its value in `field`, a node the field does not define stays
/// where the reference puts it. Called again with another `u`, it lands at
/// `X + u` again: nothing accumulates, which a restarted substep or an
/// accelerated iterate needs. The active configuration is left as it is.
/// `components` names one displacement component per axis; `None` →
/// `["ux", "uy", "uz"][..dim]`, as for [`displace`](fn@displace).
///
/// Errors: as [`Coords::displace_from`](crate::coords::Coords::displace_from),
/// and a component the field lacks.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::coords as ops_coords;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let mesh = Mesh::from_submesh(sm);
/// let current = coords.write().add_config("current");
/// let u = NodeField::new(&mesh, vec!["u_x".into(), "u_y".into()])?;
/// u.get(0)?.write().set_value(n[1].id(), "u_x", 0.5)?;
/// let names = Some(vec!["u_x".to_string(), "u_y".to_string()]);
/// ops_coords::displace_from(&u, 0, current, names)?;
/// // The reference has not moved; the current configuration has.
/// assert_eq!(n[1].position()?, vec![2.0, 0.0]);
/// coords.write().select(current)?;
/// assert_eq!(n[1].position()?, vec![2.5, 0.0]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn displace_from(
    field: &NodeField,
    reference: usize,
    target: usize,
    components: Option<Vec<String>>,
) -> Result<()> {
    let coords = field.coords()?;
    let dim = coords.read().dim() as usize;
    let comps = resolve_axis_components(field, components, dim, &["ux", "uy", "uz"])?;
    let (nodes, increments) = per_node_values(field, &comps)?;
    coords.write().displace_from(
        reference,
        target,
        nodes.iter().copied().zip(increments.chunks_exact(dim)),
    )
}

/// Run `f` with configuration `config` of `coords` active, then make the
/// configuration that was active before it active again — after an error, and
/// after a panic, too. Everything `f` computes on the geometry (matrices,
/// normals, measures) sees `config`; what it returns is handed back.
///
/// The lock on `coords` is taken on entry and on exit only: `f` reads and
/// writes the coordinates freely, and may call `in_config` again. The active
/// configuration belongs to the `Coords`, not to the caller: another thread
/// computing on the same `Coords` meanwhile sees `config` too.
///
/// Errors: `config` out of range, before `f` runs; whatever `f` returns.
///
/// ```
/// # use pyrucast::atoms::Node;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::coords as ops_coords;
/// let coords = Handle::new(Coords::new(2)?);
/// let n = Node::create_in(coords.clone(), &[1.0, 0.0])?;
/// let current = coords.write().add_config("current");
///
/// // Written in `current`, read in `current`; the reference is back after each.
/// ops_coords::in_config(&coords, current, || n.set_position(&[2.0, 0.0]))?;
/// let x = ops_coords::in_config(&coords, current, || n.position())?;
/// assert_eq!(x, vec![2.0, 0.0]);
/// assert_eq!(coords.read().active(), 0);
/// assert_eq!(n.position()?, vec![1.0, 0.0]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn in_config<R>(
    coords: &Handle<Coords>,
    config: usize,
    f: impl FnOnce() -> Result<R>,
) -> Result<R> {
    let previous = {
        let mut c = coords.write();
        let previous = c.active();
        c.select(config)?;
        previous
    };
    // Caught only to restore, then resumed: nothing observes `f`'s state after
    // a panic, so asserting it unwind-safe is sound.
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    // `previous` was valid on entry, and configurations are never removed.
    coords.write().select(previous)?;
    match out {
        Ok(result) => result,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Aggregate;
    use crate::atoms::{ElementType, Node};
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::containers::node_field::SubNodeField;
    use crate::coords::Coords;
    use crate::handle::Handle;
    use crate::ops::node_field::positions;

    #[test]
    fn set_writes_active_config() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        mesh.add_cell(&[b.id()]).unwrap();

        // Field of target positions (X/Y), round-tripped from the reader.
        let f = positions(&mesh, None).unwrap();
        {
            let mut s = f.get(0).unwrap().write();
            s.set_value(a.id(), "X", 10.0).unwrap();
            s.set_value(a.id(), "Y", 20.0).unwrap();
        }

        set(&f, None).unwrap();
        {
            let c = coords.read();
            assert_eq!(c.position(a.id()).unwrap(), &[10.0, 20.0]);
            assert_eq!(c.position(b.id()).unwrap(), &[1.0, 1.0]);
        }
    }

    #[test]
    fn displace_adds_to_active_config() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let support = Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut d = SubNodeField::from_poi1(&support, vec!["ux".into(), "uy".into()]).unwrap();
        d.set_value(a.id(), "ux", 5.0).unwrap();
        d.set_value(a.id(), "uy", -1.0).unwrap();
        d.set_value(b.id(), "ux", 2.0).unwrap();

        displace(&NodeField::from_sub(d), None).unwrap();
        {
            let c = coords.read();
            assert_eq!(c.position(a.id()).unwrap(), &[5.0, -1.0]);
            assert_eq!(c.position(b.id()).unwrap(), &[3.0, 1.0]);
        }
    }

    #[test]
    fn displace_moves_interface_nodes_once() {
        // Two zones sharing node `s`: the increment must apply once, not twice.
        let coords = Handle::new(Coords::new(1).unwrap());
        let s = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let mut mesh = Mesh::empty();
        for _ in 0..2 {
            let mut sm = SubMesh::new(coords.clone(), ElementType::POI1);
            sm.add_cell(&[s.id()]).unwrap();
            mesh.add_sub(Handle::new(sm)).unwrap();
        }
        let f = NodeField::new(&mesh, vec!["ux".into()]).unwrap();
        for i in 0..2 {
            f.get(i)
                .unwrap()
                .write()
                .set_value(s.id(), "ux", 0.5)
                .unwrap();
        }

        displace(&f, None).unwrap();
        assert_eq!(coords.read().position(s.id()).unwrap(), &[1.5]);
    }

    #[test]
    fn displace_reads_each_component_where_a_zone_holds_it() {
        // One support, `ux` in one zone and `uy` in the other: each node moves
        // by both.
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let support = Handle::new(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut ux = SubNodeField::from_poi1(&support, vec!["ux".into()]).unwrap();
        ux.set_value(a.id(), "ux", 1.0).unwrap();
        ux.set_value(b.id(), "ux", 2.0).unwrap();
        let mut uy = SubNodeField::from_poi1(&support, vec!["uy".into()]).unwrap();
        uy.set_value(a.id(), "uy", 3.0).unwrap();
        uy.set_value(b.id(), "uy", 4.0).unwrap();
        let mut f = NodeField::from_sub(ux);
        f.add_sub(Handle::new(uy)).unwrap();

        displace(&f, None).unwrap();
        let c = coords.read();
        assert_eq!(c.position(a.id()).unwrap(), &[1.0, 3.0]);
        assert_eq!(c.position(b.id()).unwrap(), &[3.0, 5.0]);
    }

    #[test]
    fn displace_refuses_a_node_lacking_a_component_and_moves_nothing() {
        // `a` holds `ux` and `uy`, `b` only `ux`.
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0, 1.0]).unwrap();
        let both = Handle::new(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
        let alone = Handle::new(SubMesh::poi1_from_nodes(std::slice::from_ref(&b)).unwrap());
        let mut f = NodeField::from_sub(
            SubNodeField::from_poi1(&both, vec!["ux".into(), "uy".into()]).unwrap(),
        );
        f.add_sub(Handle::new(
            SubNodeField::from_poi1(&alone, vec!["ux".into()]).unwrap(),
        ))
        .unwrap();
        f.add_to_component("ux", 1.0).unwrap();

        let err = displace(&f, None).unwrap_err().to_string();
        assert!(err.contains("component uy"), "{err}");
        let c = coords.read();
        assert_eq!(c.position(a.id()).unwrap(), &[0.0, 0.0]);
        assert_eq!(c.position(b.id()).unwrap(), &[1.0, 1.0]);
    }

    #[test]
    fn in_config_restores_the_previous_configuration() {
        let coords = Handle::new(Coords::new(1).unwrap());
        let a = coords.write().add_config("a");
        let b = coords.write().add_config("b");
        coords.write().select(a).unwrap();

        // After a value, nested: each level gives back the one it found.
        let seen = in_config(&coords, b, || {
            let inner = in_config(&coords, 0, || Ok(coords.read().active()))?;
            Ok((coords.read().active(), inner))
        })
        .unwrap();
        assert_eq!(seen, (b, 0));
        assert_eq!(coords.read().active(), a);

        // After an error, which is handed back.
        let err = in_config(&coords, b, || -> Result<()> {
            Err(PyrucastError::Message("inside".into()))
        });
        assert!(err.is_err());
        assert_eq!(coords.read().active(), a);

        // After a panic, which goes on unwinding.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = in_config(&coords, b, || -> Result<()> { panic!("inside") });
        }));
        assert!(panicked.is_err());
        assert_eq!(coords.read().active(), a);

        // An index out of range: nothing runs, nothing changes.
        let mut ran = false;
        assert!(in_config(&coords, 3, || {
            ran = true;
            Ok(())
        })
        .is_err());
        assert!(!ran);
        assert_eq!(coords.read().active(), a);
    }

    #[test]
    fn writers_reject_wrong_component_count() {
        let coords = Handle::new(Coords::new(2).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(coords.clone(), ElementType::POI1));
        mesh.add_cell(&[a.id()]).unwrap();
        let f = positions(&mesh, None).unwrap();
        // 2-D Coords needs 2 axis components, not 1.
        assert!(set(&f, Some(vec!["X".into()])).is_err());
        // Unknown component name.
        assert!(set(&f, Some(vec!["X".into(), "BOGUS".into()])).is_err());
    }
}
