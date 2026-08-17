//! The cubic Hermite family, checked against the one thing that can falsify it:
//! the Euler-Bernoulli stiffness matrix.
//!
//! `src/models/bernoulli.rs` carries that matrix as a **closed form** — twelve
//! numbers written by hand from the classical derivation. It is therefore an
//! oracle that owes nothing to this basis, and integrating
//!
//! ```text
//! K_ij = ∫ EI · (∂²N_i/∂x²)(∂²N_j/∂x²) dx
//! ```
//!
//! over a `HERMITE3` subspace must reproduce it to machine precision. Any error
//! in the shape functions, in their second derivatives, or in the Jacobian
//! scaling that maps a reference slope onto a physical rotation lands in this
//! comparison — which is what makes it worth more than sixteen assertions on
//! the basis alone.

use pyrucast::aggregate::Aggregate;
use pyrucast::atoms::{ElementType, Interpolation, Node};
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::coords::Coords;
use pyrucast::store::Handle;
use pyrucast::Result;

/// One `SEG2` of length `l` on the x axis, as a `HERMITE3` space.
fn segment(l: f64) -> Result<FiniteElementSpace> {
    let coords = Handle::new(Coords::new(1)?);
    let a = Node::create_in(coords.clone(), &[0.0])?;
    let b = Node::create_in(coords.clone(), &[l])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    mesh.add_cell(&[a.id(), b.id()])?;
    FiniteElementSpace::new(&mesh, Interpolation::Hermite3)
}

/// The classical Euler-Bernoulli bending stiffness on `[w_A, θ_A, w_B, θ_B]`.
fn closed_form(ei: f64, l: f64) -> [[f64; 4]; 4] {
    let c = ei / (l * l * l);
    let l2 = l * l;
    [
        [12.0 * c, 6.0 * l * c, -12.0 * c, 6.0 * l * c],
        [6.0 * l * c, 4.0 * l2 * c, -6.0 * l * c, 2.0 * l2 * c],
        [-12.0 * c, -6.0 * l * c, 12.0 * c, -6.0 * l * c],
        [6.0 * l * c, 2.0 * l2 * c, -6.0 * l * c, 4.0 * l2 * c],
    ]
}

/// Integrate `∫ EI B''ᵀ B'' dx` over the subspace, straight from the tabulated
/// reference basis — the Jacobian applied here exactly as `CellGeom` does it.
fn integrated(ei: f64, l: f64) -> Result<[[f64; 4]; 4]> {
    let fes = segment(l)?;
    let s = fes.get(0)?.read();
    let j = l / 2.0; // a straight SEG2 has a constant Jacobian
    let mut k = [[0.0_f64; 4]; 4];
    for g in 0..s.gauss_count() {
        // Reference → physical: the slope slots carry a `J`, then `J⁻²` maps
        // `∂²/∂ξ²` onto `∂²/∂x²`.
        let b: Vec<f64> = s
            .field_d2n_at_g(g)?
            .iter()
            .enumerate()
            .map(|(i, v)| (if i % 2 == 1 { v * j } else { *v }) / (j * j))
            .collect();
        let w = s.gauss_weight(g)? * j; // dx = J dξ
        for a in 0..4 {
            for c in 0..4 {
                k[a][c] += ei * b[a] * b[c] * w;
            }
        }
    }
    Ok(k)
}

/// The integrated Hermite stiffness **is** the closed-form beam matrix.
#[test]
fn the_basis_reproduces_the_euler_bernoulli_stiffness() -> Result<()> {
    for &(ei, l) in &[(1.0, 1.0), (2.1e5 * 8.3e-6, 2.5), (7.0, 0.37)] {
        let got = integrated(ei, l)?;
        let want = closed_form(ei, l);
        let scale = want[0][0].abs().max(want[1][1].abs());
        for a in 0..4 {
            for c in 0..4 {
                assert!(
                    (got[a][c] - want[a][c]).abs() <= 1e-10 * scale,
                    "EI={ei}, L={l}, K[{a}][{c}]: integrated {} vs closed form {}",
                    got[a][c],
                    want[a][c]
                );
            }
        }
    }
    Ok(())
}

/// A `HERMITE3` space carries **four** shape functions over two nodes, while its
/// geometry keeps the two Lagrange ones. That split is the whole point: the
/// element is subparametric, and a Jacobian built from four functions would be
/// meaningless.
#[test]
fn the_field_basis_and_the_geometry_part_company() -> Result<()> {
    let fes = segment(2.0)?;
    let s = fes.get(0)?.read();
    assert_eq!(s.shape_count()?, 4);
    assert_eq!(s.nodes_per_cell()?, 2);
    assert_eq!(s.field_n_at_g(0)?.len(), 4);
    assert_eq!(s.n_at_g(0)?.len(), 2, "geometry stays Lagrange-1");
    // …and the geometric basis is still a partition of unity.
    let sum: f64 = s.n_at_g(0)?.iter().sum();
    assert!((sum - 1.0).abs() < 1e-14);
    Ok(())
}

/// `HERMITE3` is defined on `SEG2` alone, and says so rather than silently
/// producing a basis of the wrong length on a triangle.
#[test]
fn hermite_is_rejected_on_anything_but_a_segment() -> Result<()> {
    let coords = Handle::new(Coords::new(2)?);
    let a = Node::create_in(coords.clone(), &[0.0, 0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0, 0.0])?;
    let c = Node::create_in(coords.clone(), &[0.0, 1.0])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
    mesh.add_cell(&[a.id(), b.id(), c.id()])?;
    let err = FiniteElementSpace::new(&mesh, Interpolation::Hermite3)
        .unwrap_err()
        .to_string();
    assert!(err.contains("HERMITE3"), "unhelpful message: {err}");
    Ok(())
}

/// A Lagrange space has no second derivatives tabulated, and refuses rather
/// than returning zeros — which would read as "this element cannot bend"
/// instead of "you asked the wrong space".
#[test]
fn a_lagrange_space_refuses_to_produce_a_curvature() -> Result<()> {
    let coords = Handle::new(Coords::new(1)?);
    let a = Node::create_in(coords.clone(), &[0.0])?;
    let b = Node::create_in(coords.clone(), &[1.0])?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    mesh.add_cell(&[a.id(), b.id()])?;
    let fes = FiniteElementSpace::lagrange1(&mesh)?;
    let s = fes.get(0)?.read();
    // The field basis falls back to the geometric one — identical, as it must be.
    assert_eq!(s.field_n_at_g(0)?, s.n_at_g(0)?);
    let err = s.field_d2n_at_g(0).unwrap_err().to_string();
    assert!(err.contains("second derivatives"), "message: {err}");
    Ok(())
}

/// Several cells in a row: the basis is tabulated once and the C¹ continuity is
/// carried by the shared `theta` degree of freedom, not by anything per-cell.
#[test]
fn a_whole_beam_mesh_can_be_hermite() -> Result<()> {
    let coords = Handle::new(Coords::new(1)?);
    let nodes: Vec<_> = (0..4)
        .map(|i| Node::create_in(coords.clone(), &[i as f64 * 0.5]))
        .collect::<Result<_>>()?;
    let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::SEG2));
    for w in nodes.windows(2) {
        mesh.add_cell(&[w[0].id(), w[1].id()])?;
    }
    let fes = FiniteElementSpace::new(&mesh, Interpolation::Hermite3)?;
    let s = fes.get(0)?.read();
    assert_eq!(s.cell_count()?, 3);
    assert_eq!(s.shape_count()?, 4);
    Ok(())
}
