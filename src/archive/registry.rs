//! The one central table: a tag in the file, back to a type in the crate.
//!
//! Serialization discovers the graph on its own, so nothing here declares what
//! points at what. This table exists only because reading has to go the other
//! way: the file says `"SubMesh"`, and something must know that means
//! [`SubMesh`](crate::containers::mesh::SubMesh).
//!
//! **One line per archivable type**, and the compiler catches a missing `Root`
//! variant. Adding a type here is the whole cost of making it archivable.

use super::{Archivable, Root};
use crate::archive::Portable;
use crate::containers::element_field::{ElementField, SubElementField};
use crate::containers::evolution::{Evolution, SubEvolution};
use crate::containers::finite_element_space::{FiniteElementSpace, SubFiniteElementSpace};
use crate::containers::matrix::{Matrix, SubMatrix};
use crate::containers::mesh::{Mesh, SubMesh};
use crate::containers::model::{Model, SubModel};
use crate::containers::node_field::{NodeField, SubNodeField};
use crate::coords::Coords;
use crate::error::{PyrucastError, Result};
use crate::handle::Handle;
use std::any::Any;

fn unknown(tag: &str) -> PyrucastError {
    PyrucastError::Message(format!(
        "archive: unknown object type \"{tag}\" — the file was written by a \
         build that knows a type this one does not"
    ))
}

fn mismatch(tag: &str) -> PyrucastError {
    PyrucastError::Message(format!(
        "archive: a record tagged \"{tag}\" does not hold a {tag} — the file is damaged"
    ))
}

/// The **shared** types: those a `Handle` inside another object can point at.
/// They travel as their own record and are resolved by identifier.
macro_rules! shared {
    ($($ty:ty => $variant:ident),* $(,)?) => {
        /// Decode one record into its object, type erased as `Handle<T>`.
        pub(crate) fn decode_node(tag: &str, bytes: &[u8]) -> Result<Box<dyn Any + Send + Sync>> {
            $(
                if tag == <$ty as Archivable>::TAG {
                    let mut value = <$ty as Portable>::from_bytes(bytes)?;
                    value.on_load();
                    return Ok(Box::new(Handle::new(value)));
                }
            )*
            Err(unknown(tag))
        }

        /// Wrap an already-decoded object as a named root.
        pub(crate) fn wrap_root(tag: &str, any: &(dyn Any + Send + Sync)) -> Result<Root> {
            $(
                if tag == <$ty as Archivable>::TAG {
                    return any
                        .downcast_ref::<Handle<$ty>>()
                        .cloned()
                        .map(Root::$variant)
                        .ok_or_else(|| mismatch(tag));
                }
            )*
            Err(unknown(tag))
        }
    };
}

shared! {
    Coords => Coords,
    SubMesh => SubMesh,
    SubFiniteElementSpace => SubFiniteElementSpace,
    SubNodeField => SubNodeField,
    SubElementField => SubElementField,
    SubEvolution => SubEvolution,
    SubModel => SubModel,
    SubMatrix => SubMatrix,
}

/// The **aggregates**: held by value, never pointed at from inside the graph,
/// so they travel inline in their root entry rather than as a shared record.
macro_rules! inline {
    ($($ty:ty => $variant:ident),* $(,)?) => {
        pub(crate) fn decode_inline(tag: &str, bytes: &[u8]) -> Result<Root> {
            $(
                if tag == <$ty as Archivable>::TAG {
                    let mut value = <$ty as Portable>::from_bytes(bytes)?;
                    value.on_load();
                    return Ok(Root::$variant(value));
                }
            )*
            Err(unknown(tag))
        }
    };
}

inline! {
    Mesh => Mesh,
    FiniteElementSpace => FiniteElementSpace,
    NodeField => NodeField,
    ElementField => ElementField,
    Evolution => Evolution,
    Model => Model,
    Matrix => Matrix,
}
