//! Assembly operators — turn a [`crate::containers::model::Model`] into a
//! [`crate::containers::matrix::Matrix`] (stiffness, mass) or
//! [`crate::containers::node_field::SubNodeField`] (RHS).
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
use crate::containers::field::SubField;
use crate::containers::matrix::{ComputedRecipe, Matrix, SubMatrix};
use crate::containers::model::{Model, SubModel};
use crate::error::{PyrucastError, Result};
use crate::models::Contribution;
use crate::store::{insert, read, Handle};

pub mod coloring;
pub mod flux;
pub mod scatter;
pub use flux::{flux, FluxDensity};

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
    let mut k = Matrix::empty();

    // Pass 1 — add every block. Each sub-model declares its
    // [`Contribution`](crate::models::Contribution)s; the assembler folds them
    // in without a per-type `match`. A `Computed` contribution (volumetric
    // physics) becomes a recipe-carrying block scattered straight into the CSR;
    // a `Literal` one (Dirichlet, any multi-block physics) carries its values.
    for sub_h in model {
        // Build the contribution(s) under a read guard, then drop it before
        // `add_sub` (which takes the store write lock).
        let built = {
            let sub = read(sub_h)?;
            // Generic over the physics: a sub-model needs material data iff it
            // declares a material FE subspace. No per-variant match.
            let material = match sub.material_fespace() {
                Some(fespace) => {
                    let m = materials.sub_for_fespace(&fespace)?;
                    if let Some(required) = sub.material_components() {
                        validate_material(&m, required)?;
                    }
                    Some(m)
                }
                None => None,
            };
            let mut blocks = Vec::new();
            for c in sub.as_kind().contributions(material.as_ref())? {
                blocks.extend(build_contribution(c, sub_h, material.clone())?);
            }
            blocks
        };
        for block in built {
            k.add_sub(insert(block))?;
        }
    }

    // Pass 2 — global assembly, injected via `set_assembled` (Option B: the
    // global assembler lives here, not in `Matrix`, so there is no
    // matrix↔kernel cycle). The CSR sparsity ([`scatter::build_pattern`]) is a
    // function of the model's block topology, so it is memoised on the model and
    // reused across assemblies; the values are scattered into it in parallel by
    // cell colour ([`scatter::scatter_parallel`]).
    let pattern = model.stiffness_pattern(|| scatter::build_pattern(&k))?;
    let csr = scatter::scatter_parallel(&k, pattern.as_ref())?;
    k.set_assembled(pattern.row_dofs.clone(), pattern.col_dofs.clone(), csr);
    Ok(k)
}

/// Assemble (or re-assemble) `k` **from its blocks alone** — no `Model`.
///
/// The sparsity is rebuilt from the current blocks ([`scatter::build_pattern`],
/// self-contained: a computed block resolves its fill through its recipe, a
/// literal block through its COO) and the values are scattered into it. This is
/// the composition path: after adding a block of any provenance to an already
/// assembled matrix — which `finalize` cannot handle once computed blocks are
/// present, and which `stiffness` cannot reach (it only knows a `Model`) — call
/// this to fold the new block in.
///
/// ```ignore
/// let mut k = assemble::stiffness(&model, &materials)?;
/// k.add_sub(insert(some_block))?;   // invalidates the assembled state
/// assemble::assemble(&mut k)?;      // re-assembles, new block included
/// ```
///
/// Unlike [`stiffness`], this does not consult the model's cached pattern (there
/// is no model here), so it rebuilds the sparsity each call — fine for the
/// occasional composition; hot repeated assembly of a fixed model should keep
/// going through [`stiffness`].
pub fn assemble(k: &mut Matrix) -> Result<()> {
    let pattern = scatter::build_pattern(k)?;
    let csr = scatter::scatter_parallel(k, &pattern)?;
    k.set_assembled(pattern.row_dofs, pattern.col_dofs, csr);
    Ok(())
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
) -> Result<Vec<SubMatrix>> {
    let mut blocks = match contribution {
        Contribution::Computed(layout) => {
            let recipe = ComputedRecipe {
                submodel: sub_h.clone(),
                fespaces: layout.fespaces,
                material,
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
        Contribution::Literal(blocks) => blocks,
    };
    // Tag every emitted block with its sub-model's physics nature — computed and
    // literal alike (so a Dirichlet C/Cᵀ pair is tagged too) — for Matrix::filter.
    let physics = read(sub_h)?.physics();
    for b in &mut blocks {
        b.set_physics(physics);
    }
    Ok(blocks)
}

/// Assemble the mass matrix `M` for `model`.
///
/// v0 stub: no physics has a mass term yet, so this returns an empty
/// finalized [`Matrix`] with the model's DOF layout. Kept alongside
/// [`stiffness`] so the assembler family lives in one place.
pub fn mass(model: &Model) -> Result<Matrix> {
    // `model` is unused until a physics introduces a mass term; binding it
    // documents the intended signature (assemble over the model's DOFs).
    let _ = model;
    let mut m = Matrix::empty();
    m.finalize()?;
    Ok(m)
}

/// Ensure `material` carries every component declared as required by
/// the physics. Errors with both lists for a clear message.
fn validate_material(material: &Handle<SubElementField>, required: &[&str]) -> Result<()> {
    let have: Vec<String> = read(material)?.components().to_vec();
    for req in required {
        if !have.iter().any(|c| c == req) {
            return Err(PyrucastError::Message(format!(
                "assemble::stiffness: required material component '{}' missing on \
                 SubElementField (has: [{}])",
                req,
                have.join(", ")
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::containers::finite_element_space::FiniteElementSpace;
    use crate::containers::mesh::{Coords, ElementType, Mesh, Node, SubMesh};
    use crate::ops::build::material_field_per_sub_model;

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
                let kind = sub.as_kind();
                let mut blocks = Vec::new();
                for c in kind.contributions(material.as_ref())? {
                    match c {
                        Contribution::Computed(_) => {
                            blocks.extend(kind.build_stiffness_blocks(material.as_ref())?);
                        }
                        Contribution::Literal(bs) => blocks.extend(bs),
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
        let multiplier = crate::ops::mesher::barycenter(&imposed).unwrap();
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
        let multiplier = crate::ops::mesher::barycenter(&imposed).unwrap();
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
    /// scatter. Lets a test drive `scatter::*` directly on the same blocks the
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
                for c in sub.as_kind().contributions(material.as_ref()).unwrap() {
                    blocks.extend(build_contribution(c, sub_h, material.clone()).unwrap());
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
        let pattern = scatter::build_pattern(&k).unwrap();
        let csr = scatter::scatter_serial(&k, &pattern).unwrap();

        let k_ref = assemble_literal_reference(&model, &materials).unwrap();
        let csr_ref = k_ref.to_csr().unwrap();

        assert_eq!(csr.row_offsets(), csr_ref.row_offsets());
        assert_eq!(csr.col_indices(), csr_ref.col_indices());
        assert_eq!(csr.values(), csr_ref.values());
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
        let pattern = scatter::build_pattern(&k).unwrap();
        let csr = scatter::scatter_serial(&k, &pattern).unwrap();

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
        assemble(&mut k).unwrap();

        let after = k.get(a.id(), "q", a.id(), "T").unwrap();
        assert!(
            (after - (before + 10.0)).abs() <= 1e-12 * (1.0 + before.abs()),
            "composition failed: before {before}, after {after}"
        );
    }

    /// A matrix carrying a computed block cannot be assembled through
    /// `finalize` — it must go through `ops::assemble`. `finalize` says so
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
}
