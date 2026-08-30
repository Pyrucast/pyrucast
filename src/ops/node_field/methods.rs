//! Méthodes de délégation — la face « sujet » des opérateurs de ce module.
//!
//! Voir `CONVENTIONS.md` § « Le verbe exposé aussi en méthode » : la fonction
//! libre reste la forme canonique et porte la documentation ; ces méthodes ne
//! contiennent aucune logique. Les `impl` vivent ici plutôt que dans
//! `containers/` — un conteneur ne doit pas dépendre d'un opérateur, et Rust
//! autorise un `impl` inhérent dans n'importe quel module de la crate de
//! définition.
//!
//! Ne sont **pas** exposés ici, faute de sens pour toute instance du type :
//! `internal_forces` et `internal_forces_continuum` (elles lisent la contrainte
//! de Voigt par nom : `sigma_xx`, `sigma_zz`, …), et `merge`, qui est
//! symétrique — `a | b` est déjà sa forme.

use crate::containers::element_field::ElementField;
use crate::containers::finite_element_space::FiniteElementSpace;
use crate::containers::mesh::Mesh;
use crate::containers::node_field::NodeField;
use crate::error::Result;
use crate::ops::node_field::FluxDensity;

impl NodeField {
    /// Voir [`node_field::consolidate`](fn@crate::ops::node_field::consolidate).
    pub fn consolidate(&self) -> Result<NodeField> {
        crate::ops::node_field::consolidate(self)
    }

    /// Voir [`node_field::restrict`](fn@crate::ops::node_field::restrict).
    pub fn restrict(&self, mesh: &Mesh) -> Result<NodeField> {
        crate::ops::node_field::restrict(self, mesh)
    }

    /// Voir [`node_field::restrict_like`](fn@crate::ops::node_field::restrict_like).
    pub fn restrict_like(&self, target: &NodeField) -> Result<NodeField> {
        crate::ops::node_field::restrict_like(self, target)
    }
}

impl ElementField {
    /// Voir [`node_field::divergence`](fn@crate::ops::node_field::divergence).
    pub fn divergence(&self) -> Result<NodeField> {
        crate::ops::node_field::divergence(self)
    }
}

impl Mesh {
    /// Voir [`node_field::positions`](fn@crate::ops::node_field::positions).
    pub fn positions(&self, components: Option<Vec<String>>) -> Result<NodeField> {
        crate::ops::node_field::positions(self, components)
    }
}

impl FiniteElementSpace {
    /// Voir [`node_field::flux`](fn@crate::ops::node_field::flux).
    pub fn flux(&self, density: FluxDensity, component: &str) -> Result<NodeField> {
        crate::ops::node_field::flux(self, density, component)
    }
}
