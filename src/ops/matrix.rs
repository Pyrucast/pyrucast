//! Operators that **produce** a
//! [`Matrix`] — the assemblers proper.
//!
//! The nodal assemblies (`flux`, `internal_forces`) share this module's
//! machinery ([`crate::ops::scatter`], [`crate::ops::coloring`]) but end on
//! a vector, so they live in [`crate::ops::node_field`].
//!
//! The per-physics integrands live in [`crate::models`]
//! (`heat_conduction`, `dirichlet`, …). This layer orchestrates the
//! loop over sub-models, the DOF layout, and boundary-condition
//! application.
//!
//! # Material lookup
//!
//! Material data is supplied as an [`ElementField`] aggregate. For every
//! sub-model that needs material values (e.g. `HeatConduction`), the
//! assembler picks the [`SubElementField`] whose `SubFiniteElementSpace`
//! handle matches the sub-model's own FE subspace. This lets each zone
//! carry its own material — different conductivities, different
//! materials — without coupling the (declarative) model to the
//! (per-iteration, mutable) material state.
//!
//! Sub-models that don't need material data (`Dirichlet`, …) are
//! independent of the supplied `ElementField`; an `ElementField`
//! covering only some of the FE subspaces is therefore valid as long as
//! every material-hungry sub-model finds its match.

use crate::aggregate::Aggregate;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::matrix::{ComputedRecipe, Matrix, SubMatrix};
use crate::containers::model::{Model, SubModel};
use crate::error::Result;
use crate::models::{Contribution, MatrixKind};
use crate::store::{insert, read, Handle};
use nalgebra_sparse::{CooMatrix, CsrMatrix};

/// Assemble the stiffness matrix `K` for `model`.
///
/// `materials` is an [`ElementField`] aggregate; each sub-model that
/// needs material data picks the [`SubElementField`] whose FE subspace
/// matches its own (zone-wise materials).
///
/// Each [`crate::containers::model::SubModel`] contributes one or more
/// [`crate::containers::matrix::SubMatrix`] blocks
/// (`HeatConduction` → 1 block, `Dirichlet` → C + Cᵀ).
/// The aggregate is finalized before being returned.
pub fn stiffness(model: &Model, materials: &ElementField) -> Result<Matrix> {
    assemble_kind(model, materials, MatrixKind::Stiffness, None)
}

/// Assemble the element matrix of a given [`MatrixKind`] for `model` — the
/// shared engine behind [`stiffness`] (and, for the state-dependent kinds, the
/// geometric-stiffness and consistent-tangent assemblers).
///
/// The whole *computed → scatter → pattern* pipeline is matrix-agnostic: it
/// only needs, per sub-model, the block layout and the per-cell kernel to drive.
/// This function selects both by `kind` (via
/// [`SubModelKind::contributions`](crate::models::SubModelKind::contributions))
/// and threads the resolved material and — for `Geometric` / `Tangent` — the
/// per-sub-model `state` sub-field into each block's
/// [`ComputedRecipe`]. `state` is an [`ElementField`] aggregate (current stress
/// / algorithmic tangent) resolved zone-wise like `materials`, or `None` for the
/// state-free kinds.
pub fn assemble_kind(
    model: &Model,
    materials: &ElementField,
    kind: MatrixKind,
    state: Option<&ElementField>,
) -> Result<Matrix> {
    let mut k = Matrix::empty();

    // Pass 1 — add every block. Each sub-model declares its
    // [`Contribution`](crate::models::Contribution)s for this `kind`; the
    // assembler folds them in without a per-type `match`. A `Computed`
    // contribution (volumetric physics) becomes a recipe-carrying block
    // scattered straight into the CSR; a `Literal` one (Dirichlet, any
    // multi-block physics — stiffness only) carries its values.
    for sub_h in model {
        // Build the contribution(s) under a read guard, then drop it before
        // `add_sub` (which takes the store write lock).
        let built = {
            let sub = read(sub_h)?;
            // Generic over the physics: a sub-model needs material data iff it
            // declares a material FE subspace. No per-variant match.
            let material = match sub.material_fespace() {
                Some(fespace) => {
                    // Resolve the material zone by the components this physics
                    // needs, so a shared fespace carrying several
                    // component-disjoint material zones (e.g. thermal `k` +
                    // mechanical `E`/`nu` on one mesh) resolves each physics'
                    // own zone without an explicit consolidate.
                    let m = match sub.material_components() {
                        Some(required) => materials.sub_for_fespace_with(&fespace, required)?,
                        None => materials.sub_for_fespace(&fespace)?,
                    };
                    Some(m)
                }
                None => None,
            };
            // Resolve the sub-model's own state sub-field (stress / tangent) the
            // same way, when a state aggregate is supplied and the physics reads
            // material on a cell fespace.
            let state_sub = match (state, sub.material_fespace()) {
                (Some(sf), Some(fespace)) => Some(sf.sub_for_fespace(&fespace)?),
                _ => None,
            };
            let mut blocks = Vec::new();
            for c in sub.as_kind().contributions(kind, material.as_ref())? {
                blocks.extend(build_contribution(
                    c,
                    sub_h,
                    material.clone(),
                    kind,
                    state_sub.clone(),
                )?);
            }
            blocks
        };
        for block in built {
            k.add_sub(insert(block))?;
        }
    }

    // Pass 2 — global assembly, injected via `set_assembled` (Option B: the
    // global assembler lives here, not in `Matrix`, so there is no
    // matrix↔kernel cycle). The CSR sparsity ([`crate::ops::scatter::build_pattern`]) is a
    // function of the model's block topology for this `kind`, so it is memoised
    // per kind on the model and reused across assemblies; the values are
    // scattered into it in parallel by cell colour ([`crate::ops::scatter::scatter_parallel`]).
    let pattern = model.matrix_pattern(kind, || crate::ops::scatter::build_pattern(&k))?;
    let csr = crate::ops::scatter::scatter_parallel(&k, pattern.as_ref())?;
    k.set_assembled(pattern.row_dofs.clone(), pattern.col_dofs.clone(), csr);
    Ok(k)
}

// The composition path lives here rather than in `containers/matrix.rs`: it
// needs the scatter machinery of `ops/`, and a container must not depend on an
// operator. Rust lets an inherent `impl` sit in any module of the defining
// crate, so `Matrix` gains the method without the container gaining the
// dependency.
impl Matrix {
    /// Assemble (or re-assemble) this matrix **from its blocks alone** — no
    /// `Model`.
    ///
    /// The sparsity is rebuilt from the current blocks
    /// ([`crate::ops::scatter::build_pattern`], self-contained: a computed
    /// block resolves its fill through its recipe, a literal block through its
    /// COO) and the values are scattered into it. This is the composition path:
    /// after adding a block of any provenance to an already assembled matrix —
    /// which [`Matrix::finalize`] cannot handle once computed blocks are
    /// present, and which [`stiffness`] cannot reach (it only knows a `Model`)
    /// — call this to fold the new block in.
    ///
    /// ```ignore
    /// let mut k = crate::ops::matrix::stiffness(&model, &materials)?;
    /// k.add_sub(insert(some_block))?;   // invalidates the assembled state
    /// k.assemble()?;                    // re-assembles, new block included
    /// ```
    ///
    /// Unlike [`stiffness`], this does not consult the model's cached pattern
    /// (there is no model here), so it rebuilds the sparsity each call — fine
    /// for the occasional composition; hot repeated assembly of a fixed model
    /// should keep going through [`stiffness`].
    pub fn assemble(&mut self) -> Result<()> {
        let pattern = crate::ops::scatter::build_pattern(self)?;
        let csr = crate::ops::scatter::scatter_parallel(self, &pattern)?;
        self.set_assembled(pattern.row_dofs, pattern.col_dofs, csr);
        Ok(())
    }
}

/// Turn one [`Contribution`] into a [`SubMatrix`] block ready to add to the
/// aggregate. A `Computed` contribution is wrapped into a recipe-carrying block
/// (material resolved by the assembler, sub-model handle threaded in); a
/// `Literal` one is passed through as-is. Yields a `Vec` because a single
/// literal contribution may carry several blocks (Dirichlet's C / Cᵀ).
fn build_contribution(
    contribution: Contribution,
    sub_h: &Handle<SubModel>,
    material: Option<Handle<SubElementField>>,
    kind: MatrixKind,
    state: Option<Handle<SubElementField>>,
) -> Result<Vec<SubMatrix>> {
    let mut blocks = match contribution {
        Contribution::Computed(layout) => {
            let recipe = ComputedRecipe {
                submodel: sub_h.clone(),
                fespaces: layout.fespaces,
                material,
                kind,
                state,
                col_fespaces: Vec::new(),
            };
            vec![SubMatrix::computed(
                layout.support.clone(),
                layout.support,
                layout.dual_vars,
                layout.primal_vars,
                layout.ordering,
                layout.symmetric,
                recipe,
            )?]
        }
        // An inter-mesh block: same computed path, but rows and columns on
        // different supports. Never symmetric on its own — only the four blocks
        // of an exchange law are, together, exactly as for Dirichlet's C / Cᵀ.
        Contribution::Coupling(layout) => {
            let recipe = ComputedRecipe {
                submodel: sub_h.clone(),
                fespaces: layout.fespaces,
                material,
                kind,
                state,
                col_fespaces: layout.col_fespaces,
            };
            vec![SubMatrix::computed(
                layout.row_support,
                layout.col_support,
                layout.dual_vars,
                layout.primal_vars,
                layout.ordering,
                false,
                recipe,
            )?]
        }
        Contribution::Literal(blocks) => blocks,
    };
    // Tag every emitted block with its sub-model's physics nature set — computed
    // and literal alike (so a Dirichlet C/Cᵀ pair is tagged too) — for
    // Matrix::filter.
    let physics = read(sub_h)?.physics().to_vec();
    for b in &mut blocks {
        b.set_physics(physics.clone());
    }
    Ok(blocks)
}

/// Assemble the consistent **mass** matrix `M` for `model` (Cast3M `MASS`), or
/// the **heat-capacity** matrix `C` for a thermal model (Cast3M `CAPA`).
///
/// Mechanics contributes `M = ∫ ρ Nᵀ N` (material `rho`); heat conduction
/// contributes `C = ∫ ρ cp Nᵀ N` (material `rho`, `cp`). A physics with no mass
/// term (a boundary Robin/convection term, or a Lagrange constraint) contributes
/// nothing. `materials` supplies the density/heat coefficients zone-wise, exactly
/// like [`stiffness`].
pub fn mass(model: &Model, materials: &ElementField) -> Result<Matrix> {
    assemble_kind(model, materials, MatrixKind::Mass, None)
}

/// Assemble the **geometric (initial-stress) stiffness** `K_g` for `model`
/// (Cast3M `KSIG`): `K_g = ∫ Gᵀ σ̂ G`, the stress-stiffening term for buckling
/// and prestress analyses.
///
/// `stress` is the current Cauchy-stress [`ElementField`] (Voigt-named
/// `sigma_*`, e.g. the output of [`crate::ops::element_field::behavior::integrate`]), resolved
/// zone-wise like `materials`; the kernel is law-independent given it. `materials`
/// is still used to resolve each mechanical zone (its `E`/`nu`).
pub fn geometric(model: &Model, materials: &ElementField, stress: &ElementField) -> Result<Matrix> {
    assemble_kind(model, materials, MatrixKind::Geometric, Some(stress))
}

/// Assemble the **consistent (algorithmic) tangent** `K_t = ∫ Bᵀ D_alg B` for
/// `model` (Cast3M `KTAN`) — the operator that gives quadratic Newton
/// convergence for a non-linear law.
///
/// `state` is the behaviour field produced by [`crate::ops::element_field::behavior::integrate`]
/// at the current iterate: besides the stress it carries the per-Gauss
/// algorithmic modulus `D_alg` (the `ktan_*` components), which this assembler
/// reads back. For a **linear** physics (elasticity) the tangent is the elastic
/// stiffness and `state` is ignored. `materials` resolves each zone like
/// [`stiffness`].
pub fn tangent(model: &Model, materials: &ElementField, state: &ElementField) -> Result<Matrix> {
    assemble_kind(model, materials, MatrixKind::Tangent, Some(state))
}

/// **Lump** an assembled matrix into a diagonal one by **row-sum concentration**
/// (Cast3M `LUMP`): each diagonal entry becomes the sum of its row, every
/// off-diagonal is dropped. Applied to a consistent mass / heat-capacity matrix
/// it yields the diagonal (lumped) mass, which conserves the total mass
/// (`Σ_i M_lumped[i,i] = Σ_ij M[i,j]`) — the cheap, decoupled form used by
/// explicit transient schemes and simple eigen estimates.
///
/// The input must be assembled and square (row and column DOFs coincide
/// position-for-position, as they do for a mass/capacity matrix). The result is
/// a new assembled [`Matrix`] with the same DOF layout; the input is untouched.
pub fn lump(m: &Matrix) -> Result<Matrix> {
    let csr = m.to_csr()?;
    let row_dofs = m.row_dofs()?;
    let col_dofs = m.col_dofs()?;
    let n = csr.nrows();
    if csr.ncols() != n {
        return Err(crate::error::PyrucastError::Message(format!(
            "lump: matrix must be square, got {}×{}",
            n,
            csr.ncols()
        )));
    }
    // Diagonal = per-row sum, assembled straight into a diagonal CSR.
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for (i, row) in csr.row_iter().enumerate() {
        let s: f64 = row.values().iter().sum();
        if s != 0.0 {
            rows.push(i);
            cols.push(i);
            vals.push(s);
        }
    }
    let coo = CooMatrix::try_from_triplets(n, n, rows, cols, vals)
        .map_err(|e| crate::error::PyrucastError::Message(format!("lump: invalid COO: {e}")))?;
    let diag = CsrMatrix::from(&coo);
    let mut out = Matrix::empty();
    out.set_assembled(row_dofs, col_dofs, diag);
    Ok(out)
}

// ─── Méthodes de délégation ────────────────────────────────────────────────
//
// Voir `CONVENTIONS.md` § « Le verbe exposé aussi en méthode ». Le nom change
// entre les deux formes : la fonction libre reçoit le qualificatif de son
// module (`matrix::stiffness`), la méthode n'en a pas et doit le porter
// (`model.stiffness_matrix`).

impl Model {
    /// Voir [`matrix::stiffness`](fn@crate::ops::matrix::stiffness).
    pub fn stiffness_matrix(&self, materials: &ElementField) -> Result<Matrix> {
        stiffness(self, materials)
    }

    /// Voir [`matrix::mass`](fn@crate::ops::matrix::mass).
    pub fn mass_matrix(&self, materials: &ElementField) -> Result<Matrix> {
        mass(self, materials)
    }

    /// Voir [`matrix::geometric`](fn@crate::ops::matrix::geometric).
    pub fn geometric_matrix(
        &self,
        materials: &ElementField,
        stress: &ElementField,
    ) -> Result<Matrix> {
        geometric(self, materials, stress)
    }

    /// Voir [`matrix::tangent`](fn@crate::ops::matrix::tangent).
    pub fn tangent_matrix(&self, materials: &ElementField, state: &ElementField) -> Result<Matrix> {
        tangent(self, materials, state)
    }
}

impl Matrix {
    /// Voir [`matrix::lump`](fn@crate::ops::matrix::lump).
    pub fn lump(&self) -> Result<Matrix> {
        lump(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::{ElementType, Node};
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Mesh, SubMesh};
    use crate::coords::Coords;
    use crate::ops::element_field::material_field_per_sub_model;

    /// Assemble the stiffness the **literal** way: every sub-model fills its
    /// `SubMatrix` blocks eagerly and `finalize` scatters them. This is the
    /// historical path the computed path must match bit-for-bit — kept here
    /// purely as the equivalence reference.
    ///
    /// It reuses the [`Contribution`] seam so it stays honest: a `Literal`
    /// contribution (Dirichlet's C / Cᵀ) is taken as-is, while a `Computed` one
    /// is materialised through its physics' `build_stiffness_blocks` — the very
    /// literal kernel the scatter path is being checked against.
    fn assemble_literal_reference(model: &Model, materials: &ElementField) -> Result<Matrix> {
        let mut k = Matrix::empty();
        for sub_h in model {
            let blocks = {
                let sub = read(sub_h)?;
                let material = match sub.material_fespace() {
                    Some(fespace) => Some(materials.sub_for_fespace(&fespace)?),
                    None => None,
                };
                let phys = sub.as_kind();
                let mut blocks = Vec::new();
                for c in phys.contributions(MatrixKind::Stiffness, material.as_ref())? {
                    match c {
                        Contribution::Computed(_) => {
                            blocks.extend(phys.build_stiffness_blocks(material.as_ref())?);
                        }
                        Contribution::Literal(bs) => blocks.extend(bs),
                        // The literal path exists as the bit-for-bit equivalence
                        // reference of the *computed* single-mesh path; an
                        // inter-mesh block has no such counterpart to compare to.
                        Contribution::Coupling(_) => {
                            return Err(crate::error::PyrucastError::Message(format!(
                                "{}: an inter-mesh coupling block has no literal form — \
                                 use the computed path (ops::matrix::stiffness)",
                                phys.label()
                            )))
                        }
                    }
                }
                blocks
            };
            for block in blocks {
                k.add_sub(insert(block))?;
            }
        }
        k.finalize()?;
        Ok(k)
    }

    /// Two heat-conduction zones sharing a node, plus a Dirichlet constraint —
    /// so the assembly mixes **computed** blocks (the two zones) with **literal**
    /// ones (Dirichlet's C / Cᵀ), and a shared node forces accumulation.
    fn two_zone_heat_with_dirichlet() -> (Model, ElementField) {
        let coords = insert(Coords::new(1).unwrap());
        let n0 = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let n1 = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let n2 = Node::create_in(coords.clone(), &[2.0]).unwrap();
        let mut mesh = Mesh::empty();
        let sm_a = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
            sm.add_cell(&[n0.id(), n1.id()]).unwrap();
            insert(sm)
        };
        let sm_b = {
            let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
            sm.add_cell(&[n1.id(), n2.id()]).unwrap();
            insert(sm)
        };
        mesh.add_sub(sm_a).unwrap();
        mesh.add_sub(sm_b).unwrap();
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(insert(
                SubModel::heat_conduction(fes.get(0).unwrap()).unwrap(),
            ))
            .unwrap();
        let imposed =
            Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&n0)).unwrap());
        let multiplier = crate::ops::mesh::barycenter(&imposed).unwrap();
        model
            .add_sub(insert(
                SubModel::dirichlet(
                    "T".into(),
                    "q".into(),
                    &imposed,
                    &multiplier,
                    None,
                    None,
                    Default::default(),
                )
                .unwrap(),
            ))
            .unwrap();
        model
            .add_sub(insert(
                SubModel::heat_conduction(fes.get(1).unwrap()).unwrap(),
            ))
            .unwrap();

        let materials =
            material_field_per_sub_model(&model, &[&[("k", 1.0)], &[], &[("k", 4.0)]]).unwrap();
        (model, materials)
    }

    /// A single heat-conduction zone over an `n_elems`-element SEG2 chain, plus
    /// a Dirichlet constraint at the left end. Interior nodes are shared by two
    /// cells, so the zone needs a real (two-colour) cell colouring — the
    /// parallel scatter genuinely reorders per-slot summation, unlike a
    /// one-cell-per-block mesh.
    fn chain_heat_with_dirichlet(n_elems: usize) -> (Model, ElementField) {
        let coords = insert(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..=n_elems)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
        for i in 0..n_elems {
            sm.add_cell(&[nodes[i].id(), nodes[i + 1].id()]).unwrap();
        }
        let mesh = Mesh::from_submesh(sm);
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(insert(
                SubModel::heat_conduction(fes.get(0).unwrap()).unwrap(),
            ))
            .unwrap();
        let imposed =
            Mesh::from_submesh(SubMesh::poi1_from_nodes(std::slice::from_ref(&nodes[0])).unwrap());
        let multiplier = crate::ops::mesh::barycenter(&imposed).unwrap();
        model
            .add_sub(insert(
                SubModel::dirichlet(
                    "T".into(),
                    "q".into(),
                    &imposed,
                    &multiplier,
                    None,
                    None,
                    Default::default(),
                )
                .unwrap(),
            ))
            .unwrap();

        let materials = material_field_per_sub_model(&model, &[&[("k", 2.0)], &[]]).unwrap();
        (model, materials)
    }

    /// A Timoshenko beam over an `n_elems`-element SEG2 chain (1-D). Exercises the
    /// **multi-fespace** computed path: each block integrates two FE subspaces
    /// (bending full Gauss + shear reduced) sharing one mesh, and interior nodes
    /// are shared so the parallel scatter genuinely colours the cells.
    fn timoshenko_beam(n_elems: usize) -> (Model, ElementField) {
        let coords = insert(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..=n_elems)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
        for i in 0..n_elems {
            sm.add_cell(&[nodes[i].id(), nodes[i + 1].id()]).unwrap();
        }
        let mesh = Mesh::from_submesh(sm);
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();

        let mut model = Model::empty();
        model
            .add_sub(insert(SubModel::timoshenko(fes.get(0).unwrap()).unwrap()))
            .unwrap();
        let materials = material_field_per_sub_model(
            &model,
            &[&[("E", 3.0), ("I", 2.0), ("G", 5.0), ("A_s", 2.0)]],
        )
        .unwrap();
        (model, materials)
    }

    /// Pass 1 of [`stiffness`] alone: the block aggregate (computed blocks for
    /// the volumetric zones, literal for the rest) **before** the global
    /// scatter. Lets a test drive `crate::ops::scatter::*` directly on the same blocks the
    /// real assembler builds.
    fn assemble_computed_blocks(model: &Model, materials: &ElementField) -> Matrix {
        let mut k = Matrix::empty();
        for sub_h in model {
            let built = {
                let sub = read(sub_h).unwrap();
                let material = sub
                    .material_fespace()
                    .map(|fespace| materials.sub_for_fespace(&fespace).unwrap());
                let mut blocks = Vec::new();
                for c in sub
                    .as_kind()
                    .contributions(MatrixKind::Stiffness, material.as_ref())
                    .unwrap()
                {
                    blocks.extend(
                        build_contribution(c, sub_h, material.clone(), MatrixKind::Stiffness, None)
                            .unwrap(),
                    );
                }
                blocks
            };
            for b in built {
                k.add_sub(insert(b)).unwrap();
            }
        }
        k
    }

    /// The **serial** scatter (`build_pattern` + `scatter_serial`) reproduces the
    /// literal reference **bit-for-bit**: same sparsity and same values to the
    /// last bit (it accumulates each slot in the triplet-stream order).
    #[test]
    fn serial_scatter_equals_literal_bit_for_bit() {
        let (model, materials) = chain_heat_with_dirichlet(6);
        let k = assemble_computed_blocks(&model, &materials);
        let pattern = crate::ops::scatter::build_pattern(&k).unwrap();
        let csr = crate::ops::scatter::scatter_serial(&k, &pattern).unwrap();

        let k_ref = assemble_literal_reference(&model, &materials).unwrap();
        let csr_ref = k_ref.to_csr().unwrap();

        assert_eq!(csr.row_offsets(), csr_ref.row_offsets());
        assert_eq!(csr.col_indices(), csr_ref.col_indices());
        assert_eq!(csr.values(), csr_ref.values());
    }

    /// `SubMatrix::factor` scales every entry the **serial** scatter emits, on
    /// both branches: the computed heat-conduction blocks and the literal
    /// Dirichlet C/Cᵀ block (`assemble_computed_blocks` mixes both, per its doc).
    #[test]
    fn scatter_serial_applies_factor_to_computed_and_literal_blocks() {
        let (model, materials) = chain_heat_with_dirichlet(6);
        let k = assemble_computed_blocks(&model, &materials);
        let pattern = crate::ops::scatter::build_pattern(&k).unwrap();
        let csr_unscaled = crate::ops::scatter::scatter_serial(&k, &pattern).unwrap();

        let scaled = (&k * 3.0).unwrap();
        let pattern_scaled = crate::ops::scatter::build_pattern(&scaled).unwrap();
        let csr_scaled = crate::ops::scatter::scatter_serial(&scaled, &pattern_scaled).unwrap();

        assert_eq!(csr_scaled.row_offsets(), csr_unscaled.row_offsets());
        assert_eq!(csr_scaled.col_indices(), csr_unscaled.col_indices());
        for (x, y) in csr_scaled.values().iter().zip(csr_unscaled.values()) {
            assert!(
                (x - 3.0 * y).abs() <= 1e-9 * (1.0 + y.abs()),
                "value mismatch: {x} vs 3×{y}"
            );
        }
    }

    /// The **parallel** colour-driven scatter (`stiffness`) matches the literal
    /// reference to floating tolerance — the sparsity is identical, the values
    /// agree up to the reordered summation the colouring induces (so *not*
    /// bit-for-bit with the serial path, by construction).
    #[test]
    fn parallel_scatter_matches_literal_within_tol() {
        let (model, materials) = chain_heat_with_dirichlet(6);
        let k_new = stiffness(&model, &materials).unwrap();
        let k_ref = assemble_literal_reference(&model, &materials).unwrap();
        let csr_new = k_new.to_csr().unwrap();
        let csr_ref = k_ref.to_csr().unwrap();

        assert_eq!(csr_new.row_offsets(), csr_ref.row_offsets());
        assert_eq!(csr_new.col_indices(), csr_ref.col_indices());
        for (x, y) in csr_new.values().iter().zip(csr_ref.values()) {
            assert!(
                (x - y).abs() <= 1e-12 * (1.0 + y.abs()),
                "value mismatch: {x} vs {y}"
            );
        }
    }

    /// Scaling the real production `stiffness()` output (computed volumetric
    /// blocks + literal Dirichlet block) with `Matrix * f64`, then re-assembling
    /// with [`Matrix::assemble`] (required: `finalize` refuses a computed
    /// block), matches the literal reference scaled by hand.
    #[test]
    fn scaled_stiffness_matches_scaled_literal_reference() {
        let (model, materials) = chain_heat_with_dirichlet(6);
        let k = stiffness(&model, &materials).unwrap();
        let mut scaled = (&k * 2.5).unwrap();
        assert!(
            scaled.finalize().is_err(),
            "finalize must still refuse a computed block after scaling"
        );
        scaled.assemble().unwrap();

        let k_ref = assemble_literal_reference(&model, &materials).unwrap();
        let csr_new = scaled.to_csr().unwrap();
        let csr_ref = k_ref.to_csr().unwrap();
        assert_eq!(csr_new.row_offsets(), csr_ref.row_offsets());
        assert_eq!(csr_new.col_indices(), csr_ref.col_indices());
        for (x, y) in csr_new.values().iter().zip(csr_ref.values()) {
            assert!(
                (x - 2.5 * y).abs() <= 1e-9 * (1.0 + y.abs()),
                "value mismatch: {x} vs 2.5×{y}"
            );
        }
    }

    /// The colouring is fixed and each colour's writes are disjoint, so the
    /// parallel scatter is reproducible: two assemblies of the same model give
    /// bit-for-bit identical values (independent of thread scheduling).
    #[test]
    fn parallel_scatter_is_deterministic() {
        let (model, materials) = chain_heat_with_dirichlet(6);
        let a = stiffness(&model, &materials).unwrap();
        let b = stiffness(&model, &materials).unwrap();
        assert_eq!(a.to_csr().unwrap().values(), b.to_csr().unwrap().values());
    }

    /// A **multi-fespace** element (Timoshenko: bending full-Gauss + shear
    /// reduced, two subspaces over one mesh) assembles through the computed
    /// serial scatter **bit-for-bit** identically to its literal reference — the
    /// two `CellGeom` per cell produce the exact same triplet stream as the
    /// single-block driver.
    #[test]
    fn timoshenko_multi_fespace_serial_equals_literal_bit_for_bit() {
        let (model, materials) = timoshenko_beam(5);
        let k = assemble_computed_blocks(&model, &materials);
        let pattern = crate::ops::scatter::build_pattern(&k).unwrap();
        let csr = crate::ops::scatter::scatter_serial(&k, &pattern).unwrap();

        let k_ref = assemble_literal_reference(&model, &materials).unwrap();
        let csr_ref = k_ref.to_csr().unwrap();

        assert_eq!(csr.row_offsets(), csr_ref.row_offsets());
        assert_eq!(csr.col_indices(), csr_ref.col_indices());
        assert_eq!(csr.values(), csr_ref.values());
    }

    /// Same Timoshenko beam through the **parallel** colour-driven scatter (the
    /// real `stiffness` path): identical sparsity, values within tolerance.
    #[test]
    fn timoshenko_multi_fespace_parallel_matches_literal_within_tol() {
        let (model, materials) = timoshenko_beam(5);
        let k_new = stiffness(&model, &materials).unwrap();
        let k_ref = assemble_literal_reference(&model, &materials).unwrap();
        let csr_new = k_new.to_csr().unwrap();
        let csr_ref = k_ref.to_csr().unwrap();

        assert_eq!(csr_new.row_offsets(), csr_ref.row_offsets());
        assert_eq!(csr_new.col_indices(), csr_ref.col_indices());
        for (x, y) in csr_new.values().iter().zip(csr_ref.values()) {
            assert!(
                (x - y).abs() <= 1e-12 * (1.0 + y.abs()),
                "timoshenko value mismatch: {x} vs {y}"
            );
        }
    }

    /// The sparsity pattern is memoised on the model and reused across
    /// assemblies — a second assembly with **different materials** hits the
    /// cache yet must still produce the correct values (the pattern is
    /// material-independent, the values are not).
    #[test]
    fn cached_pattern_reused_across_materials() {
        let (model, _) = chain_heat_with_dirichlet(6);
        // First assembly (cache miss) with k = 2.
        let m1 = material_field_per_sub_model(&model, &[&[("k", 2.0)], &[]]).unwrap();
        let _ = stiffness(&model, &m1).unwrap();
        // Second assembly (cache hit) with k = 7 on the same model.
        let m2 = material_field_per_sub_model(&model, &[&[("k", 7.0)], &[]]).unwrap();
        let k2 = stiffness(&model, &m2).unwrap();

        let k_ref = assemble_literal_reference(&model, &m2).unwrap();
        let csr2 = k2.to_csr().unwrap();
        let csr_ref = k_ref.to_csr().unwrap();
        assert_eq!(csr2.row_offsets(), csr_ref.row_offsets());
        assert_eq!(csr2.col_indices(), csr_ref.col_indices());
        for (x, y) in csr2.values().iter().zip(csr_ref.values()) {
            assert!(
                (x - y).abs() <= 1e-12 * (1.0 + y.abs()),
                "cached-pattern reassembly value mismatch: {x} vs {y}"
            );
        }
    }

    /// Composition: after `stiffness`, add a **literal block of arbitrary
    /// provenance** to the (computed-block) matrix and re-assemble it with the
    /// self-contained [`assemble`]. `finalize` refuses (computed blocks present),
    /// but `assemble` rebuilds the sparsity from the blocks and folds the new
    /// contribution in.
    #[test]
    fn assemble_composes_extra_literal_block() {
        // One heat element on nodes a—b.
        let coords = insert(Coords::new(1).unwrap());
        let a = Node::create_in(coords.clone(), &[0.0]).unwrap();
        let b = Node::create_in(coords.clone(), &[1.0]).unwrap();
        let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
        sm.add_cell(&[a.id(), b.id()]).unwrap();
        let mesh = Mesh::from_submesh(sm);
        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let mut model = Model::empty();
        model
            .add_sub(insert(
                SubModel::heat_conduction(fes.get(0).unwrap()).unwrap(),
            ))
            .unwrap();
        let materials = material_field_per_sub_model(&model, &[&[("k", 1.0)]]).unwrap();

        let mut k = stiffness(&model, &materials).unwrap();
        let before = k.get(a.id(), "q", a.id(), "T").unwrap();

        // A hand-built literal block (no model behind it) adding +10 at (a,q)×(a,T).
        let support = insert(SubMesh::poi1_from_nodes(std::slice::from_ref(&a)).unwrap());
        let mut blk = SubMatrix::new(
            support.clone(),
            support,
            vec!["q".into()],
            vec!["T".into()],
            crate::containers::matrix::DofOrdering::NodesThenVars,
            true,
        )
        .unwrap();
        blk.add_entry(a.id(), "q", a.id(), "T", 10.0).unwrap();
        k.add_sub(insert(blk)).unwrap();

        // `finalize` can't (computed block present); the self-contained path can.
        assert!(k.finalize().is_err());
        k.assemble().unwrap();

        let after = k.get(a.id(), "q", a.id(), "T").unwrap();
        assert!(
            (after - (before + 10.0)).abs() <= 1e-12 * (1.0 + before.abs()),
            "composition failed: before {before}, after {after}"
        );
    }

    /// A matrix carrying a computed block cannot be assembled through
    /// `finalize` — it must go through `ops::matrix`. `finalize` says so
    /// rather than silently dropping the computed contribution.
    #[test]
    fn finalize_rejects_computed_block() {
        let (model, materials) = two_zone_heat_with_dirichlet();
        // Build the blocks (computed for the HC zones) but stop before the
        // global assembly, then try to finalize directly.
        let mut k = Matrix::empty();
        for sub_h in &model {
            let built = {
                let sub = read(sub_h).unwrap();
                let material = sub
                    .material_fespace()
                    .map(|fespace| materials.sub_for_fespace(&fespace).unwrap());
                sub.as_kind().stiffness_layout().map(|layout| {
                    SubMatrix::computed(
                        layout.support.clone(),
                        layout.support,
                        layout.dual_vars,
                        layout.primal_vars,
                        layout.ordering,
                        layout.symmetric,
                        ComputedRecipe {
                            submodel: sub_h.clone(),
                            fespaces: layout.fespaces,
                            material,
                            kind: MatrixKind::Stiffness,
                            state: None,
                            col_fespaces: Vec::new(),
                        },
                    )
                    .unwrap()
                })
            };
            if let Some(block) = built {
                k.add_sub(insert(block)).unwrap();
            }
        }
        assert!(k.finalize().is_err());
    }

    /// The support a stiffness block carries (its `col_support`) is the cached
    /// `to_poi1` of its fespace submesh — the very slot `restrict` onto the same
    /// mesh lands on. So a solve/`K·x` output and a `restrict` onto that mesh
    /// share one support and combine directly by the field operators.
    #[test]
    fn stiffness_support_matches_restrict_onto_the_same_mesh() {
        use crate::aggregate::Aggregate;
        use crate::containers::node_field::{NodeField, SubNodeField};
        use crate::ops::node_field::restrict;

        let coords = insert(Coords::new(1).unwrap());
        let nodes: Vec<Node> = (0..=2)
            .map(|i| Node::create_in(coords.clone(), &[i as f64]).unwrap())
            .collect();
        let mut sm = SubMesh::new(coords.clone(), ElementType::SEG2);
        sm.add_cell(&[nodes[0].id(), nodes[1].id()]).unwrap();
        sm.add_cell(&[nodes[1].id(), nodes[2].id()]).unwrap();
        let mesh = Mesh::from_submesh(sm);

        let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
        let mut model = Model::empty();
        model
            .add_sub(insert(
                SubModel::heat_conduction(fes.get(0).unwrap()).unwrap(),
            ))
            .unwrap();
        let materials = material_field_per_sub_model(&model, &[&[("k", 1.0)]]).unwrap();
        let k = stiffness(&model, &materials).unwrap();

        // A field restricted onto the very same mesh.
        let mut psm = SubMesh::new(coords.clone(), ElementType::POI1);
        for nd in &nodes {
            psm.add_cell(&[nd.id()]).unwrap();
        }
        let f =
            NodeField::from_sub(SubNodeField::from_poi1(&insert(psm), vec!["T".into()]).unwrap());
        let r = restrict(&f, &mesh).unwrap();

        let col = read(k.iter().next().unwrap())
            .unwrap()
            .col_support()
            .clone();
        let rsup = read(&r.get(0).unwrap()).unwrap().support();
        assert!(
            col.same_slot(&rsup),
            "block support and restrict onto the same mesh share one slot"
        );
    }
}
