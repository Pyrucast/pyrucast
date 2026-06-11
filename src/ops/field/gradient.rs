//! Gradient of a nodal field at the Gauss points — purely geometric, no
//! model/physics involved (it depends only on the FE space and the field).
//!
//! `∇f = Σ_i f_i ∇N_i` evaluated cell-by-cell, Gauss-point by Gauss-point.
//! The shared [`subspace_gradients`] core also backs
//! [`crate::ops::field::deformation`] (the symmetric gradient).

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::field::Field;
use crate::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use crate::containers::mesh::NodeId;
use crate::containers::node_field::{FieldSnapshot, NodeField};
use crate::error::Result;
use crate::store::{insert, with, Handle};

/// Axis suffixes for spatial directions (`x`, `y`, `z`).
pub(crate) const AXES: [&str; 3] = ["x", "y", "z"];

/// Per-`(cell, Gauss)` partial derivatives `∂(component)/∂x_axis` of a node
/// field's selected components on **one** FE subspace. Flat, row-major,
/// indexed `((cell * n_g + g) * n_comp + ci) * space_dim + a`.
pub(crate) struct Gradients {
    pub(crate) fespace: Handle<SubFiniteElementSpace>,
    pub(crate) n_cells: usize,
    pub(crate) n_g: usize,
    pub(crate) space_dim: usize,
    pub(crate) n_comp: usize,
    values: Vec<f64>,
}

impl Gradients {
    /// `∂(component ci)/∂x_a` at `(cell, gauss)`.
    pub(crate) fn at(&self, cell: usize, g: usize, ci: usize, a: usize) -> f64 {
        self.values[((cell * self.n_g + g) * self.n_comp + ci) * self.space_dim + a]
    }
}

/// Compute `∂(component)/∂x_axis` for each of `components` of `field` at
/// every Gauss point of `fespace` (one FE subspace). `∇f = Σ_i f_i ∇N_i`.
/// `field` is a zone snapshot: node lookups resolve across the zones
/// (first zone defining the pair wins) and error if a cell node lacks one
/// of `components`.
pub(crate) fn subspace_gradients(
    fespace: &Handle<SubFiniteElementSpace>,
    field: &FieldSnapshot,
    components: &[String],
) -> Result<Gradients> {
    // Snapshot the reference/geometry data in one critical section
    // (∇N at every (cell, Gauss) + connectivity), mirroring the assembler.
    struct Snap {
        n_cells: usize,
        n_g: usize,
        space_dim: usize,
        n_nodes: usize,
        conn: Vec<NodeId>,
        dn_dx: Vec<Vec<Vec<f64>>>, // [cell][g][i * space_dim + a]
    }
    let snap = with(fespace, |s| -> Result<Snap> {
        let n_cells = s.cell_count()?;
        let n_nodes = s.nodes_per_cell()?;
        let n_g = s.gauss_count();
        let space_dim = s.space_dim();
        let submesh = s.submesh();
        let conn: Vec<NodeId> = with(&submesh, |sm| sm.connectivity().to_vec())?;
        let mut dn_dx = Vec::with_capacity(n_cells);
        for cell in 0..n_cells {
            let mut per_g = Vec::with_capacity(n_g);
            for g in 0..n_g {
                per_g.push(s.dn_dx(cell, g)?);
            }
            dn_dx.push(per_g);
        }
        Ok(Snap { n_cells, n_g, space_dim, n_nodes, conn, dn_dx })
    })??;

    let n_comp = components.len();
    let mut values = vec![0.0; snap.n_cells * snap.n_g * n_comp * snap.space_dim];
    for cell in 0..snap.n_cells {
        let ids = &snap.conn[cell * snap.n_nodes..(cell + 1) * snap.n_nodes];
        for g in 0..snap.n_g {
            for (ci, comp) in components.iter().enumerate() {
                for a in 0..snap.space_dim {
                    let mut grad = 0.0;
                    for i in 0..snap.n_nodes {
                        grad += field.value(ids[i], comp)?
                            * snap.dn_dx[cell][g][i * snap.space_dim + a];
                    }
                    let idx = ((cell * snap.n_g + g) * n_comp + ci) * snap.space_dim + a;
                    values[idx] = grad;
                }
            }
        }
    }

    Ok(Gradients {
        fespace: fespace.clone(),
        n_cells: snap.n_cells,
        n_g: snap.n_g,
        space_dim: snap.space_dim,
        n_comp,
        values,
    })
}

/// Gradient `∇f` of a node `field` at the Gauss points of every subspace of
/// `fespace`.
///
/// Geometric and physics-agnostic: each component of `field` is
/// differentiated w.r.t. every spatial axis, giving an [`ElementField`] with
/// one component `grad_<name>_<axis>` per (input component, axis) pair, in
/// order component-major then axis (`grad_T_x`, …). Feed the result to
/// [`crate::ops::behavior::integrate`] as the deformation input of a physics
/// whose behaviour consumes a gradient (heat conduction).
pub fn gradient(field: &NodeField, fespace: &FiniteElementSpace) -> Result<ElementField> {
    let components = Field::components(field)?;
    let snapshot = field.snapshot()?;
    let mut out = ElementField::empty();
    for sub in fespace {
        let g = subspace_gradients(sub, &snapshot, &components)?;
        out.add_sub(insert(gradients_to_field(&g, &components)?))?;
    }
    Ok(out)
}

/// Lay the raw [`Gradients`] out as a `grad_<comp>_<axis>` [`SubElementField`].
fn gradients_to_field(g: &Gradients, components: &[String]) -> Result<SubElementField> {
    let mut names = Vec::with_capacity(g.n_comp * g.space_dim);
    for comp in components {
        for a in 0..g.space_dim {
            names.push(format!("grad_{comp}_{}", AXES[a]));
        }
    }
    let mut field = SubElementField::new(g.fespace.clone(), names)?;
    for cell in 0..g.n_cells {
        for gp in 0..g.n_g {
            let mut out_c = 0;
            for ci in 0..g.n_comp {
                for a in 0..g.space_dim {
                    field.set(cell, gp, out_c, g.at(cell, gp, ci, a))?;
                    out_c += 1;
                }
            }
        }
    }
    Ok(field)
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::mesh::{Configuration, ElementType, Mesh, Node, SubMesh};
    use crate::containers::node_field::SubNodeField;
    use crate::store::insert;

    /// 1-D linear field `T(x) = x` on a single SEG2 ⇒ `∇T = 1`.
    #[test]
    fn gradient_of_linear_field_on_seg2() {
        let cfg = insert(Configuration::new(1).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::SEG2));
        mesh.add_cell(&[a.id(), b.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support = insert(SubMesh::poi1_from_nodes(&[a.clone(), b.clone()]).unwrap());
        let mut t = SubNodeField::from_poi1(&support, vec!["T".into()]).unwrap();
        t.set_value(a.id(), "T", 0.0).unwrap();
        t.set_value(b.id(), "T", 2.0).unwrap(); // T = x
        let t = NodeField::from_sub(t);

        let grad = gradient(&t, &fes).unwrap();
        assert_eq!(grad.len(), 1);
        with(&grad.get(0).unwrap(), |s| {
            assert_eq!(s.components(), &["grad_T_x".to_string()]);
            for g in 0..s.gauss_count() {
                assert!((s.value(0, g, "grad_T_x").unwrap() - 1.0).abs() < 1e-12);
            }
        })
        .unwrap();
    }

    /// 2-D linear field `f = 2x + 3y` on a TRI3 ⇒ `∇f = (2, 3)`.
    #[test]
    fn gradient_2d_linear_field_on_tri3() {
        let cfg = insert(Configuration::new(2).unwrap());
        let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
        let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();
        let c = Node::create_in(cfg.clone(), &[0.0, 1.0]).unwrap();
        let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::TRI3));
        mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let support =
            insert(SubMesh::poi1_from_nodes(&[a.clone(), b.clone(), c.clone()]).unwrap());
        let mut f = SubNodeField::from_poi1(&support, vec!["f".into()]).unwrap();
        f.set_value(a.id(), "f", 0.0).unwrap(); // 2·0 + 3·0
        f.set_value(b.id(), "f", 2.0).unwrap(); // 2·1 + 3·0
        f.set_value(c.id(), "f", 3.0).unwrap(); // 2·0 + 3·1
        let f = NodeField::from_sub(f);

        let grad = gradient(&f, &fes).unwrap();
        with(&grad.get(0).unwrap(), |s| {
            assert_eq!(s.components(), &["grad_f_x".to_string(), "grad_f_y".to_string()]);
            for g in 0..s.gauss_count() {
                assert!((s.value(0, g, "grad_f_x").unwrap() - 2.0).abs() < 1e-12);
                assert!((s.value(0, g, "grad_f_y").unwrap() - 3.0).abs() < 1e-12);
            }
        })
        .unwrap();
    }
}
