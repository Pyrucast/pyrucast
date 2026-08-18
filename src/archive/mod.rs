//! Saving and reloading a graph of objects.
//!
//! Writing an object writes everything it needs, and reading gives them back
//! **sharing what they shared**: two fields carried by one support come back as
//! two fields carried by one support, not by two copies of it.
//!
//! ```no_run
//! use pyrucast::archive;
//! # use pyrucast::containers::mesh::Mesh;
//! # fn demo(mesh: &Mesh, temperature: &pyrucast::containers::node_field::NodeField)
//! # -> pyrucast::error::Result<()> {
//! archive::save("etude.pyr", &[
//!     ("maillage fin", mesh as &dyn archive::ArchiveRoot),
//!     ("T (°C)",       temperature),
//!     ("pas de temps", &0.05_f64),
//! ])?;
//!
//! let mut objets = archive::load("etude.pyr")?;
//! let mesh2 = objets.mesh("maillage fin")?;
//! # Ok(())
//! # }
//! ```
//!
//! # What the file carries, and what it does not
//!
//! The rule is **recomputable ⇒ not written**. Every cache and every memoised
//! table stays out: the assembled CSR, the solver's factorization, the cell
//! colouring, the lazy index maps, a field's copy of its support's
//! connectivity. What comes back rebuilds them on demand, exactly as a
//! freshly-built graph would.
//!
//! Reference counts are not written either. Object lifetimes are counted by the
//! `Arc` inside each [`Handle`], which counts itself as
//! `load` hands the handles out. Node counts inside a
//! [`Coords`] are **recounted from zero**: each reloaded
//! sub-mesh re-increments the nodes it uses. The consequence is deliberate —
//! *reloaded nodes are protected only by the objects present in the file*.
//! Saving a bare `Coords` and reloading it gives nodes at count zero, which a
//! `gc()` will collect, exactly as if the same objects had been rebuilt by hand
//! without keeping a `Node`. A [`Node`](crate::atoms::Node) is an atom of the
//! caller's stack; it is not archived.
//!
//! # Reloading adds, it never replaces
//!
//! There is no global registry: [`load`] builds fresh objects and hands back
//! their handles. Whatever already lives in the session is untouched.
//!
//! # This is a session format, not an exchange format
//!
//! The file is pyrucast's own, versioned, and **breakable before 1.0.0** — an
//! unknown version is refused, never converted. To hand results to another tool,
//! use [`export_vtk`](crate::ops::export), which exists for that.

pub mod portable;
mod registry;
mod scope;

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
pub use portable::Portable;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::BTreeMap;
use std::path::Path;

/// First eight bytes of every archive.
const MAGIC: &[u8; 8] = b"PYRUCAST";
/// Format number. Bumped on any incompatible change; before 1.0.0 no migration
/// is offered, an unknown number is simply refused.
const FORMAT: u32 = 1;

// ─── What a type must offer to be archived ──────────────────────────────────

/// A type that can be a node of a saved graph.
///
/// [`Portable`] supplies the byte contract — bincode, normalized little-endian,
/// `usize` on 64 bits, identical on Linux and Windows. This trait adds the two
/// things serde cannot express: a **tag** so a record can be dispatched back to
/// its type on reload, and a **hook** to rebuild what the file deliberately does
/// not carry.
///
/// Nothing here declares which objects an implementor points at: the walk is the
/// serialization itself: `Handle::serialize` interns what it meets. Adding a
/// `Handle` field to a
/// type therefore adds an edge, with nothing to keep in step.
pub trait Archivable: Portable + Any + Send + Sync {
    /// Name of the type in the file. Must be unique, and stable as long as the
    /// format number is.
    const TAG: &'static str;

    /// Called on a freshly decoded object, before anyone can see it.
    ///
    /// The place to rebuild what is recomputable, and the only one.
    fn on_load(&mut self) {}
}

// ─── Simple values ──────────────────────────────────────────────────────────

/// A value that is not a pyrucast object: the time step, the name of a load
/// case, the list of instants.
///
/// Lists are **homogeneous**. Nested lists and dictionaries are out of scope.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    Bools(Vec<bool>),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Texts(Vec<String>),
}

// ─── What comes back ────────────────────────────────────────────────────────

/// One named entry of a reloaded archive.
///
/// The set is **closed**, which is the point: a `match` on it is exhaustive, so
/// a type added to the archive cannot be silently forgotten at the boundary —
/// the Python binding included.
pub enum Root {
    Coords(Handle<Coords>),
    Mesh(Mesh),
    SubMesh(Handle<SubMesh>),
    FiniteElementSpace(FiniteElementSpace),
    SubFiniteElementSpace(Handle<SubFiniteElementSpace>),
    NodeField(NodeField),
    SubNodeField(Handle<SubNodeField>),
    ElementField(ElementField),
    SubElementField(Handle<SubElementField>),
    Evolution(Evolution),
    SubEvolution(Handle<SubEvolution>),
    Model(Model),
    SubModel(Handle<SubModel>),
    Matrix(Matrix),
    SubMatrix(Handle<SubMatrix>),
    /// A simple value rather than a pyrucast object.
    Value(Value),
}

impl Root {
    /// Name of the kind held, for error messages: `"Mesh"`, `"float"`, …
    pub fn type_name(&self) -> &'static str {
        match self {
            Root::Coords(_) => Coords::TAG,
            Root::Mesh(_) => Mesh::TAG,
            Root::SubMesh(_) => SubMesh::TAG,
            Root::FiniteElementSpace(_) => FiniteElementSpace::TAG,
            Root::SubFiniteElementSpace(_) => SubFiniteElementSpace::TAG,
            Root::NodeField(_) => NodeField::TAG,
            Root::SubNodeField(_) => SubNodeField::TAG,
            Root::ElementField(_) => ElementField::TAG,
            Root::SubElementField(_) => SubElementField::TAG,
            Root::Evolution(_) => Evolution::TAG,
            Root::SubEvolution(_) => SubEvolution::TAG,
            Root::Model(_) => Model::TAG,
            Root::SubModel(_) => SubModel::TAG,
            Root::Matrix(_) => Matrix::TAG,
            Root::SubMatrix(_) => SubMatrix::TAG,
            Root::Value(v) => match v {
                Value::Bool(_) => "bool",
                Value::Int(_) => "int",
                Value::Float(_) => "float",
                Value::Text(_) => "str",
                Value::Bools(_) => "list[bool]",
                Value::Ints(_) => "list[int]",
                Value::Floats(_) => "list[float]",
                Value::Texts(_) => "list[str]",
            },
        }
    }
}

impl std::fmt::Debug for Root {
    /// Type and, for a shared object, its short tag — never the contents. A
    /// reloaded mesh may hold millions of cells.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Root::Coords(h) => write!(f, "{h}"),
            Root::SubMesh(h) => write!(f, "{h}"),
            Root::SubFiniteElementSpace(h) => write!(f, "{h}"),
            Root::SubNodeField(h) => write!(f, "{h}"),
            Root::SubElementField(h) => write!(f, "{h}"),
            Root::SubEvolution(h) => write!(f, "{h}"),
            Root::SubModel(h) => write!(f, "{h}"),
            Root::SubMatrix(h) => write!(f, "{h}"),
            Root::Value(v) => write!(f, "{v:?}"),
            other => write!(f, "<{}>", other.type_name()),
        }
    }
}

/// The named objects a [`load`] gave back.
///
/// Dereferences to a `BTreeMap`, so `keys()`, iteration and `len()` are
/// available — and sorted, which is also what makes an archive byte-reproducible.
/// The typed accessors ([`mesh`](Objects::mesh), …) exist to give an error that
/// names the key, the type expected and the type found.
#[derive(Default, Debug)]
pub struct Objects(BTreeMap<String, Root>);

impl std::ops::Deref for Objects {
    type Target = BTreeMap<String, Root>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Objects {
    fn wrong(&self, key: &str, expected: &str) -> PyrucastError {
        match self.0.get(key) {
            Some(r) => PyrucastError::Message(format!(
                "archive: \"{key}\" is a {}, not a {expected}",
                r.type_name()
            )),
            None => PyrucastError::Message(format!(
                "archive: no entry named \"{key}\" (available: {})",
                self.0.keys().cloned().collect::<Vec<_>>().join(", ")
            )),
        }
    }
}

/// Generates one typed accessor per variant: `objects.mesh("k")?`.
macro_rules! accessors {
    ($($name:ident, $variant:ident, $ty:ty, $label:literal;)*) => {
        impl Objects {
            $(
                #[doc = concat!("The `", $label, "` named `key`, or an error naming what was found instead.")]
                pub fn $name(&self, key: &str) -> Result<$ty> {
                    match self.0.get(key) {
                        Some(Root::$variant(v)) => Ok(v.clone()),
                        _ => Err(self.wrong(key, $label)),
                    }
                }
            )*
        }
    };
}

accessors! {
    coords, Coords, Handle<Coords>, "Coords";
    submesh, SubMesh, Handle<SubMesh>, "SubMesh";
    subfespace, SubFiniteElementSpace, Handle<SubFiniteElementSpace>, "SubFiniteElementSpace";
    sub_node_field, SubNodeField, Handle<SubNodeField>, "SubNodeField";
    sub_element_field, SubElementField, Handle<SubElementField>, "SubElementField";
    sub_evolution, SubEvolution, Handle<SubEvolution>, "SubEvolution";
    sub_model, SubModel, Handle<SubModel>, "SubModel";
    sub_matrix, SubMatrix, Handle<SubMatrix>, "SubMatrix";
}

/// Generates the accessors for the aggregates, which come back by value and are
/// therefore **taken** out of the map rather than cloned.
macro_rules! take_accessors {
    ($($name:ident, $variant:ident, $ty:ty, $label:literal;)*) => {
        impl Objects {
            $(
                #[doc = concat!("The `", $label, "` named `key`, removed from the map (it is held by value).")]
                pub fn $name(&mut self, key: &str) -> Result<$ty> {
                    match self.0.get(key) {
                        Some(Root::$variant(_)) => match self.0.remove(key) {
                            Some(Root::$variant(v)) => Ok(v),
                            _ => unreachable!("just matched"),
                        },
                        _ => Err(self.wrong(key, $label)),
                    }
                }
            )*
        }
    };
}

take_accessors! {
    mesh, Mesh, Mesh, "Mesh";
    fespace, FiniteElementSpace, FiniteElementSpace, "FiniteElementSpace";
    node_field, NodeField, NodeField, "NodeField";
    element_field, ElementField, ElementField, "ElementField";
    evolution, Evolution, Evolution, "Evolution";
    model, Model, Model, "Model";
    matrix, Matrix, Matrix, "Matrix";
}

/// Accessors for the simple values.
macro_rules! value_accessors {
    ($($name:ident, $variant:ident, $ty:ty, $label:literal;)*) => {
        impl Objects {
            $(
                #[doc = concat!("The `", $label, "` named `key`.")]
                pub fn $name(&self, key: &str) -> Result<$ty> {
                    match self.0.get(key) {
                        Some(Root::Value(Value::$variant(v))) => Ok(v.clone()),
                        _ => Err(self.wrong(key, $label)),
                    }
                }
            )*
        }
    };
}

value_accessors! {
    bool, Bool, bool, "bool";
    int, Int, i64, "int";
    float, Float, f64, "float";
    text, Text, String, "str";
    bools, Bools, Vec<bool>, "list[bool]";
    ints, Ints, Vec<i64>, "list[int]";
    floats, Floats, Vec<f64>, "list[float]";
    texts, Texts, Vec<String>, "list[str]";
}

// ─── What can be handed to `save` ───────────────────────────────────────────

/// How a named entry is written. Implementation detail of the file format.
#[doc(hidden)]
#[derive(Serialize, Deserialize)]
pub enum Entry {
    /// A shared object: the identifier of its record.
    Object(u32),
    /// An aggregate, held by value and carried inline — nothing points at it.
    Inline { tag: String, payload: Vec<u8> },
    /// A simple value.
    Value(Value),
}

/// Anything that can be handed to [`save`] under a name.
///
/// Implemented for every handle to an archivable object, for the seven
/// aggregates, and for the simple values. Sealed in practice: the method is
/// hidden, since it speaks the file format.
pub trait ArchiveRoot {
    #[doc(hidden)]
    fn to_entry(&self) -> Result<Entry>;
}

impl<T: Archivable> ArchiveRoot for Handle<T> {
    fn to_entry(&self) -> Result<Entry> {
        Ok(Entry::Object(scope::intern_handle(self)?))
    }
}

/// The aggregates travel inline: they are held by value, and no handle inside
/// the graph ever points at one.
macro_rules! inline_roots {
    ($($ty:ty),* $(,)?) => {
        $(
            impl ArchiveRoot for $ty {
                fn to_entry(&self) -> Result<Entry> {
                    Ok(Entry::Inline {
                        tag: <$ty as Archivable>::TAG.to_string(),
                        payload: self.to_bytes()?,
                    })
                }
            }
        )*
    };
}

inline_roots!(
    Mesh,
    FiniteElementSpace,
    NodeField,
    ElementField,
    Evolution,
    Model,
    Matrix,
);

macro_rules! value_roots {
    ($($ty:ty => $variant:ident),* $(,)?) => {
        $(
            impl ArchiveRoot for $ty {
                fn to_entry(&self) -> Result<Entry> {
                    Ok(Entry::Value(Value::$variant(self.clone())))
                }
            }
        )*
    };
}

value_roots!(
    bool => Bool,
    i64 => Int,
    f64 => Float,
    String => Text,
    Vec<bool> => Bools,
    Vec<i64> => Ints,
    Vec<f64> => Floats,
    Vec<String> => Texts,
);

impl ArchiveRoot for str {
    fn to_entry(&self) -> Result<Entry> {
        Ok(Entry::Value(Value::Text(self.to_string())))
    }
}

// ─── The file ───────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct Body {
    /// Informative only: never tested, only shown when something goes wrong.
    crate_version: String,
    /// `(identifier, tag, bytes)`, in dependency order.
    records: Vec<(u32, String, Vec<u8>)>,
    roots: BTreeMap<String, Entry>,
}

/// Write `roots` and everything they need to `path`.
///
/// Objects shared by several roots are written **once**; that is where the
/// no-duplication guarantee is kept. Saving twice the same objects produces the
/// same bytes.
///
/// Fails, rather than looping, if the graph has a cycle — only recomputable
/// caches may point backwards, and caches are not written.
pub fn save<P: AsRef<Path>>(path: P, roots: &[(&str, &dyn ArchiveRoot)]) -> Result<()> {
    let (root_map, records) = scope::with_write(|| {
        // Sorted first, so the identifiers are handed out in a deterministic
        // order and the file is byte-reproducible.
        let mut sorted: Vec<_> = roots.iter().collect();
        sorted.sort_by_key(|(name, _)| *name);

        let mut map: BTreeMap<String, Entry> = BTreeMap::new();
        for (name, root) in sorted {
            if map.contains_key(*name) {
                return Err(PyrucastError::Message(format!(
                    "archive: \"{name}\" is given twice"
                )));
            }
            map.insert((*name).to_string(), root.to_entry()?);
        }
        Ok(map)
    })?;

    let body = Body {
        crate_version: crate::VERSION.to_string(),
        records: records
            .into_iter()
            .map(|r| (r.id, r.tag.to_string(), r.payload))
            .collect(),
        roots: root_map,
    };

    let mut out = Vec::with_capacity(4096);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT.to_le_bytes());
    out.extend_from_slice(&body.to_bytes()?);
    std::fs::write(path.as_ref(), &out)
        .map_err(|e| PyrucastError::Io(format!("writing {}: {e}", path.as_ref().display())))
}

/// Read back what [`save`] wrote, preserving what was shared.
///
/// The objects are **new**: nothing already alive in the session is touched.
pub fn load<P: AsRef<Path>>(path: P) -> Result<Objects> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| PyrucastError::Io(format!("reading {}: {e}", path.as_ref().display())))?;

    if bytes.len() < 12 || &bytes[..8] != MAGIC {
        return Err(PyrucastError::Message(format!(
            "{} is not a pyrucast archive (bad signature)",
            path.as_ref().display()
        )));
    }
    let format = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if format != FORMAT {
        return Err(PyrucastError::Message(format!(
            "{} is in archive format {format}, this build reads format {FORMAT}. \
             Before version 1.0.0 no conversion is offered.",
            path.as_ref().display()
        )));
    }
    let body = Body::from_bytes(&bytes[12..])?;

    scope::with_read(|| {
        for (id, tag, payload) in &body.records {
            let object = registry::decode_node(tag, payload)?;
            scope::publish(*id, tag.clone(), object)?;
        }
        let mut out = BTreeMap::new();
        for (name, entry) in body.roots {
            let root = match entry {
                Entry::Value(v) => Ok(Root::Value(v)),
                Entry::Object(id) => {
                    let tag = scope::tag_of(id)?;
                    scope::with_object(id, |any| registry::wrap_root(&tag, any))?
                }
                Entry::Inline { tag, payload } => registry::decode_inline(&tag, &payload),
            }?;
            out.insert(name, root);
        }
        Ok(Objects(out))
    })
}
