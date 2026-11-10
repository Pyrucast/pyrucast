//! The values of a field along a line, as an [`Evolution`] — the curve of a
//! component against the distance travelled along a path of points.

use crate::aggregate::Aggregate;
use crate::containers::evolution::{Evolution, OutOfRange, SubEvolution, SubValue};
use crate::containers::field::Field;
use crate::containers::mesh::Mesh;
use crate::containers::node_field::NodeField;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;

/// The values of `field` along `line`: one curve per component, the
/// value against the distance travelled along the line.
///
/// `line` is a POI1 mesh, read in the order of its cells, zone after zone:
/// that order is the path. The abscissa of a point is the length of the
/// path up to it — the sum of the straight distances from one point to the
/// next, in the active configuration —, `0` at the first point. Its
/// ordinate is the value of the component at that node; a node the field
/// does not define reads zero, as a nodal field always does.
///
/// Each component of `components` gives one [`SubEvolution`] of scalars,
/// in that order, its ordinate type the component's name. To follow a
/// field of another mesh, or a field at the Gauss points, bring it onto
/// the line's points first with
/// [`project_node`](fn@crate::ops::node_field::project_node) or
/// [`project_element`](fn@crate::ops::node_field::project_element).
///
/// Errors: `line` is not POI1; two consecutive points of `line` coincide
/// (the abscissa would repeat); a component `field` does not carry;
/// `field` and `line` on different `Coords`.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::Node;
/// # use pyrucast::ops::evolution;
/// # use pyrucast::containers::field::SubField;
/// # use pyrucast::containers::node_field::NodeField;
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # use pyrucast::ops::mesh;
/// // Three points along a broken line: 3 then 4 long.
/// let coords = Handle::new(Coords::new(2)?);
/// let n: Vec<Node> = [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]]
///     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// let ligne = mesh::poi1_from_nodes(&n)?;
/// let t = NodeField::new(&ligne, vec!["T".into(), "q".into()])?;
/// for (node, v) in n.iter().zip([10.0, 40.0, 80.0]) {
///     t.get(0)?.write().set_value(node.id(), "T", v)?;
/// }
/// let le_long = evolution::along(&t, &ligne, &["T", "q"])?;
/// assert_eq!(le_long.len(), 2); // one curve per component
/// let curve = le_long.get(0)?.read().scalar_series()?;
/// assert_eq!(curve, vec![(0.0, 10.0), (3.0, 40.0), (7.0, 80.0)]);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
pub fn along(field: &NodeField, line: &Mesh, components: &[&str]) -> Result<Evolution> {
    let carried = Field::components(field);
    for c in components {
        if !carried.iter().any(|have| have == c) {
            return Err(PyrucastError::Message(format!(
                "along: the field carries no component `{c}` (it has {carried:?})"
            )));
        }
    }
    let line_coords = line.coords()?;
    if !line_coords.same_object(&field.coords()?) {
        return Err(PyrucastError::Message(
            "along: `field` and `line` hang from different Coords — bring the field onto \
             the line's points first (`project`)"
                .into(),
        ));
    }

    // The path: the points of `line` in order, and the distance to each.
    let mut nodes = Vec::new();
    for sm in line {
        let sm = sm.read();
        if sm.element_type() != crate::atoms::ElementType::POI1 {
            return Err(PyrucastError::Message(format!(
                "along: `line` must be a POI1 mesh — its points, in order — not {}",
                sm.element_type()
            )));
        }
        nodes.extend_from_slice(sm.connectivity());
    }
    let mut abscissas = Vec::with_capacity(nodes.len());
    if let Some(&first) = nodes.first() {
        let c = line_coords.read();
        let mut before = c.position(first)?;
        let mut travelled = 0.0;
        abscissas.push(travelled);
        for (k, &nid) in nodes.iter().enumerate().skip(1) {
            let here = c.position(nid)?;
            let step = here
                .iter()
                .zip(before)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            if step == 0.0 {
                return Err(PyrucastError::Message(format!(
                    "along: points {} and {k} of `line` coincide — the abscissa would \
                     repeat",
                    k - 1
                )));
            }
            travelled += step;
            abscissas.push(travelled);
            before = here;
        }
    }

    // Every value at once, node-major, then one curve per component.
    let wanted: Vec<String> = components.iter().map(|c| c.to_string()).collect();
    let view = field.view()?;
    let reads = view.resolve_reads(&wanted);
    let n_comp = wanted.len();
    let mut values = vec![0.0; nodes.len() * n_comp];
    view.gather_cell(&nodes, &reads, &mut values);
    let mut out = Evolution::default();
    for (k, name) in wanted.into_iter().enumerate() {
        let samples = abscissas
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, SubValue::Scalar(values[i * n_comp + k])))
            .collect();
        let sub = SubEvolution::new(samples, OutOfRange::Error)?.with_ordinate_type(Some(name))?;
        out.add_sub(Handle::new(sub))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::mesh::SubMesh;
    use crate::coords::Coords;

    /// Points of a fresh 3-D `Coords`, as one POI1 line.
    fn line_of(at: &[[f64; 3]]) -> (Mesh, Vec<Node>) {
        let coords = Handle::new(Coords::new(3).unwrap());
        let nodes: Vec<Node> = at
            .iter()
            .map(|p| Node::create_in(coords.clone(), p).unwrap())
            .collect();
        (crate::ops::mesh::poi1_from_nodes(&nodes).unwrap(), nodes)
    }

    /// A field linear in x along a slanted straight line: the curve is the
    /// same line against the distance travelled, each component on its own
    /// curve, all at the same abscissas.
    #[test]
    fn along_a_straight_line_gives_one_curve_per_component() {
        let at: Vec<[f64; 3]> = (0..5)
            .map(|i| [i as f64, 2.0 * i as f64, 2.0 * i as f64])
            .collect();
        let (line, nodes) = line_of(&at);
        let field = NodeField::new(&line, vec!["a".into(), "b".into()]).unwrap();
        for (node, p) in nodes.iter().zip(&at) {
            let mut z = field.get(0).unwrap().write();
            z.set_value(node.id(), "a", 1.0 + 2.0 * p[0]).unwrap();
            z.set_value(node.id(), "b", -p[0]).unwrap();
        }
        let e = along(&field, &line, &["b", "a"]).unwrap();
        assert_eq!(e.len(), 2);
        let b = e.get(0).unwrap().read();
        let a = e.get(1).unwrap().read();
        assert_eq!(
            (b.ordinate_type(), a.ordinate_type()),
            (Some("b"), Some("a"))
        );
        assert_eq!(a.abscissas(), b.abscissas());
        // Each step is √(1 + 4 + 4) = 3 long.
        for (i, (s, v)) in a.scalar_series().unwrap().into_iter().enumerate() {
            assert!((s - 3.0 * i as f64).abs() < 1e-12);
            assert!((v - (1.0 + 2.0 * i as f64)).abs() < 1e-12);
        }
    }

    #[test]
    fn along_refuses_what_has_no_path() {
        let (line, nodes) = line_of(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let field = NodeField::new(&line, vec!["a".into()]).unwrap();
        // A component the field does not carry.
        let err = along(&field, &line, &["z"]).unwrap_err();
        assert!(format!("{err}").contains("`z`"), "unexpected: {err}");
        // Two consecutive points at the same place.
        let twice = crate::ops::mesh::poi1_from_nodes(&[nodes[0].clone(), nodes[0].clone()]);
        let err = along(&field, &twice.unwrap(), &["a"]).unwrap_err();
        assert!(format!("{err}").contains("coincide"), "unexpected: {err}");
        // A line of segments is not a list of points.
        let mut sm = SubMesh::new(line.coords().unwrap(), ElementType::SEG2);
        sm.add_cell(&[nodes[0].id(), nodes[1].id()]).unwrap();
        let segments = Mesh::from_submesh(sm);
        let err = along(&field, &segments, &["a"]).unwrap_err();
        assert!(format!("{err}").contains("POI1"), "unexpected: {err}");
        // A line of another `Coords`.
        let (elsewhere, _) = line_of(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let err = along(&field, &elsewhere, &["a"]).unwrap_err();
        assert!(format!("{err}").contains("Coords"), "unexpected: {err}");
    }
}
