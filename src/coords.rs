//! Coords — node coordinates with garbage collection.
//!
//! A [`Coords`] holds **one or more configurations** (sets of coordinates)
//! for the same set of nodes, in a fixed dimension.
//!
//! # Node identity
//!
//! Every created node receives a **stable** internal identifier
//! ([`NodeId`]), unique for the lifetime of the `Coords`: **no id
//! is ever reused**, even after garbage collection. Other objects (meshes,
//! fields) can therefore reference a node by id without worrying about
//! stability.
//!
//! # Deletion policy: no direct removal
//!
//! There is **no** `remove_node` method. A referenced node is protected.
//! Only the garbage collector [`Coords::gc`] reclaims nodes whose
//! **internal** refcount has reached 0.
//!
//! # Two-level refcount model
//!
//! - The **Coords object** is kept alive by the usual
//!   [`crate::handle::Handle`] refcount.
//! - **Each node** inside the Coords has its own refcount,
//!   manipulated via [`Coords::incref`] / [`Coords::decref`]
//!   (used by [`crate::atoms::Node`] and, later, by meshes and fields).
//!
//! # Identity vs solver ordering
//!
//! An optional permutation (`Vec<u32>`) separates the **solver order**
//! from the **identity**: `permutation[node_id]` is the solver-order index
//! assigned to `node_id`. It is set by the caller today; a bandwidth-reducing
//! renumbering (Cuthill–McKee) will compute it. The identity (`NodeId`) is
//! never modified either way.
//!
//! # Multiple configurations
//!
//! Useful for switching between reference / deformed / predicted
//! configurations. An active configuration is designated by index;
//! [`Coords::position`] reads from the active one.
//!
//! # Coordinate frame
//!
//! A `Coords` also declares **how its coordinates are to be read**: plain
//! Cartesian (the default, [`Coords::new`]) or **axisymmetric**
//! ([`Coords::axisymmetric`]) — a 2-D meridian plane `(r, z)` describing a body
//! of revolution. The frame is a property of the geometry, so every mesh, FE
//! space and integral built on top of it inherits it; see [`CoordinateFrame`].
//!
//! # Example
//!
//! ```
//! use pyrucast::atoms::NodeId;
//! use pyrucast::coords::Coords;
//! use pyrucast::handle::Handle;
//!
//! let h = Handle::new(Coords::new(2).unwrap());
//! let a: NodeId = h.write().add_node(&[0.0, 0.0]).unwrap();
//! // add_node initializes refcount = 1: without decref, the node is protected.
//! assert_eq!(h.write().gc(), 0);
//! // After decref, refcount drops to 0 and gc collects it.
//! let mut c = h.write();
//! c.decref(a).unwrap();
//! assert_eq!(c.gc(), 1);
//! ```

use crate::atoms::NodeId;
use crate::error::{PyrucastError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// How the coordinates of a [`Coords`] are to be read — the geometric
/// hypothesis every integral built on top of it obeys.
///
/// This is the Cast3M `OPTI MODE` axis: it belongs to the **geometry**, not to
/// any one physics, because it changes the integration measure `dΩ` itself —
/// stiffness, mass, distributed flux, volumes and internal forces alike.
///
/// ```
/// # use pyrucast::aggregate::Aggregate;
/// # use pyrucast::atoms::{ElementType, Node};
/// # use pyrucast::containers::mesh::{Mesh, SubMesh};
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// # let coords = Handle::new(Coords::new(2).unwrap());
/// # let n: Vec<Node> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
/// #     .iter().map(|p| Node::create_in(coords.clone(), p).unwrap()).collect();
/// # let mut sm = SubMesh::new(coords.clone(), ElementType::TRI3);
/// # sm.add_cell(&[n[0].id(), n[1].id(), n[2].id()]).unwrap();
/// # let maillage = Mesh::from_submesh(sm);
/// # use pyrucast::coords::CoordinateFrame;
/// // Le repère appartient à la **géométrie**, non à une physique : c'est
/// // lui qui change la mesure d'intégration dΩ, donc raideur, masse, flux
/// // réparti, volumes et forces internes à la fois.
/// assert_eq!(Coords::new(2)?.frame(), CoordinateFrame::Cartesian);
/// let axi = Coords::axisymmetric()?;
/// assert_eq!(axi.frame(), CoordinateFrame::Axisymmetric);
/// assert!(axi.is_axisymmetric());
/// assert_eq!(axi.dim(), 2); // `x = r`, `y = z`
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinateFrame {
    /// Plain Cartesian coordinates in `dim` dimensions, `dΩ = |J| dξ`.
    #[default]
    Cartesian,
    /// 2-D meridian plane of a body of revolution: `x = r` (radius, `≥ 0`),
    /// `y = z` (axis of revolution). Integrals run over the **full ring**,
    /// `dΩ = 2πr |J| dξ`, so masses, volumes and nodal resultants are those of
    /// the whole revolved part.
    Axisymmetric,
}

impl CoordinateFrame {
    /// Whether this frame is [`Axisymmetric`](Self::Axisymmetric).
    ///
    /// ```
    /// # use pyrucast::coords::CoordinateFrame;
    /// assert!(CoordinateFrame::Axisymmetric.is_axisymmetric());
    /// assert!(!CoordinateFrame::Cartesian.is_axisymmetric());
    /// ```
    pub fn is_axisymmetric(self) -> bool {
        self == Self::Axisymmetric
    }
}

impl fmt::Display for CoordinateFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cartesian => "cartesian",
            Self::Axisymmetric => "axisymmetric",
        })
    }
}

/// Node coordinates with stable identity, multiple configurations,
/// optional solver permutation, and a garbage collector for unreferenced
/// nodes.
///
/// ```
/// # use pyrucast::coords::Coords;
/// # use pyrucast::handle::Handle;
/// // Des positions à **identité stable** : un nœud garde son `NodeId` quoi
/// // qu'il advienne des tableaux. S'y ajoutent plusieurs configurations,
/// // une permutation pour le solveur, et un ramasse-miettes.
/// let c = Handle::new(Coords::new(2)?);
/// let id = c.write().add_node(&[1.0, 2.0])?;
/// assert_eq!(c.read().position(id)?, vec![1.0, 2.0]);
/// assert_eq!(c.read().node_count(), 1);
/// // Le compteur de références protège le nœud tant qu'on le tient.
/// assert_eq!(c.write().gc(), 0);
/// c.write().decref(id)?;
/// assert_eq!(c.write().gc(), 1);
/// # Ok::<(), pyrucast::PyrucastError>(())
/// ```
#[derive(Serialize, Deserialize)]
pub struct Coords {
    dim: u8,
    /// Geometric hypothesis the coordinates obey. `#[serde(default)]` so a
    /// `Coords` serialised before the frame existed reloads as Cartesian.
    #[serde(default)]
    frame: CoordinateFrame,
    /// `configs[c][id * dim + k]` — each configuration holds `capacity * dim` values.
    configs: Vec<Vec<f64>>,
    config_names: Vec<String>,
    active: usize,
    /// `alive[id] == false` ⇒ collected by the GC. Once `false`, stays so forever.
    alive: Vec<bool>,
    /// Per-node refcount. The GC collects `alive` nodes whose refcount is 0.
    ///
    /// **Never archived.** A file holds some of the objects that reference these
    /// nodes, not all of them, so a saved count would be a count of a world that
    /// no longer exists. It is recounted from zero at reload: `on_load` zeroes
    /// it, then each reloaded `SubMesh` re-increments what it uses.
    #[serde(skip)]
    refcount: Vec<u32>,
    /// Solver permutation (length == capacity) or `None` for identity.
    permutation: Option<Vec<u32>>,
}

impl Coords {
    /// Create an empty **Cartesian** `Coords` in dimension `dim` (≥ 1). A first
    /// configuration named `"default"` is created automatically. For a body of
    /// revolution, see [`Coords::axisymmetric`].
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let plan = Coords::new(2).unwrap();
    /// assert_eq!(plan.dim(), 2);
    /// assert!(Coords::new(0).is_err()); // la dimension doit être ≥ 1
    /// ```
    pub fn new(dim: u8) -> Result<Self> {
        if dim == 0 {
            return Err(PyrucastError::Message("dim must be ≥ 1".into()));
        }
        Ok(Self {
            dim,
            frame: CoordinateFrame::Cartesian,
            configs: vec![Vec::new()],
            config_names: vec!["default".into()],
            active: 0,
            alive: Vec::new(),
            refcount: Vec::new(),
            permutation: None,
        })
    }

    /// Create an empty **axisymmetric** `Coords` — the 2-D meridian plane of a
    /// body of revolution, `x = r` (radius, `≥ 0`) and `y = z` (axis). The
    /// dimension is necessarily 2, so it is not an argument.
    ///
    /// Every FE space built over this geometry integrates over the full ring
    /// (`dΩ = 2πr |J| dξ`); mechanics additionally gains the hoop strain
    /// `ε_θθ = u_r / r` through
    /// [`Kinematics::Axisymmetric`](crate::models::tensor::Kinematics::Axisymmetric).
    ///
    /// ```
    /// use pyrucast::coords::Coords;
    ///
    /// let c = Coords::axisymmetric().unwrap();
    /// assert_eq!(c.dim(), 2);
    /// assert!(c.is_axisymmetric());
    /// ```
    pub fn axisymmetric() -> Result<Self> {
        Ok(Self {
            frame: CoordinateFrame::Axisymmetric,
            ..Self::new(2)?
        })
    }

    /// Geometric dimension.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// assert_eq!(Coords::new(3).unwrap().dim(), 3);
    /// ```
    pub fn dim(&self) -> u8 {
        self.dim
    }

    /// Geometric hypothesis these coordinates obey.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::coords::CoordinateFrame;
    /// assert_eq!(Coords::new(2).unwrap().frame(), CoordinateFrame::Cartesian);
    /// assert_eq!(Coords::axisymmetric().unwrap().frame(), CoordinateFrame::Axisymmetric);
    /// ```
    pub fn frame(&self) -> CoordinateFrame {
        self.frame
    }

    /// Whether these coordinates describe a body of revolution — the shorthand
    /// for `frame().is_axisymmetric()`.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// assert!(!Coords::new(2).unwrap().is_axisymmetric());
    /// assert!(Coords::axisymmetric().unwrap().is_axisymmetric());
    /// ```
    pub fn is_axisymmetric(&self) -> bool {
        self.frame.is_axisymmetric()
    }

    /// Reject a negative radius in an axisymmetric frame (`x = r ≥ 0`), where
    /// `what` names the calling operation.
    fn check_radius(&self, what: &str, coords: &[f64]) -> Result<()> {
        if self.frame.is_axisymmetric() && coords[0] < 0.0 {
            return Err(PyrucastError::Message(format!(
                "{what}: negative radius x = {} in an axisymmetric Coords \
                 (x = r ≥ 0, y = z along the axis of revolution)",
                coords[0]
            )));
        }
        Ok(())
    }

    /// Number of live (not collected) nodes.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// c.add_node(&[0.0, 0.0]).unwrap();
    /// assert_eq!(c.node_count(), 1); // nœuds **vivants**, pas la capacité
    /// ```
    pub fn node_count(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }

    /// Capacity (total slots, live + collected). Never decreases.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// let id = c.add_node(&[0.0, 0.0]).unwrap();
    /// c.decref(id).unwrap();
    /// c.gc();
    /// // Le nœud est mort mais la place reste réservée : capacité ≠ node_count.
    /// assert_eq!((c.node_count(), c.capacity()), (0, 1));
    /// ```
    pub fn capacity(&self) -> usize {
        self.alive.len()
    }

    /// Whether a node is still alive.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// let id = c.add_node(&[0.0, 0.0]).unwrap();
    /// assert!(c.is_alive(id));
    /// c.decref(id).unwrap();
    /// c.gc();
    /// assert!(!c.is_alive(id));
    /// ```
    pub fn is_alive(&self, id: NodeId) -> bool {
        self.alive.get(id.0 as usize).copied().unwrap_or(false)
    }

    /// Add a node with these coordinates in **all** configurations. Initializes its
    /// refcount to 1 — the caller is responsible for at least one decrement
    /// (typically through the end-of-life of a [`crate::atoms::Node`]).
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// let id = c.add_node(&[1.0, 2.0]).unwrap();
    /// assert_eq!(c.position(id).unwrap(), &[1.0, 2.0]);
    /// assert_eq!(c.refcount(id), 1); // créé avec un ticket
    /// // La longueur doit valoir la dimension.
    /// assert!(c.add_node(&[0.0]).is_err());
    /// ```
    pub fn add_node(&mut self, coords: &[f64]) -> Result<NodeId> {
        if coords.len() != self.dim as usize {
            return Err(PyrucastError::Message(format!(
                "add_node: expected {} coordinates, got {}",
                self.dim,
                coords.len()
            )));
        }
        self.check_radius("add_node", coords)?;
        let id = self.alive.len() as u32;
        for set in &mut self.configs {
            set.extend_from_slice(coords);
        }
        self.alive.push(true);
        self.refcount.push(1);
        if let Some(perm) = &mut self.permutation {
            perm.push(id);
        }
        Ok(NodeId(id))
    }

    /// Increment the refcount of a live node.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// let id = c.add_node(&[0.0, 0.0]).unwrap();
    /// c.incref(id).unwrap();
    /// assert_eq!(c.refcount(id), 2);
    /// ```
    pub fn incref(&mut self, id: NodeId) -> Result<()> {
        self.ensure_alive(id)?;
        let r = &mut self.refcount[id.0 as usize];
        *r = r.saturating_add(1);
        Ok(())
    }

    /// Decrement the refcount of a live node. The node is not immediately
    /// collected even if the refcount reaches 0: call
    /// [`Coords::gc`] for that.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// let id = c.add_node(&[0.0, 0.0]).unwrap();
    /// c.decref(id).unwrap();
    /// assert_eq!(c.refcount(id), 0); // plus personne ne le tient : gc peut le prendre
    /// ```
    pub fn decref(&mut self, id: NodeId) -> Result<()> {
        self.ensure_alive(id)?;
        let r = &mut self.refcount[id.0 as usize];
        if *r == 0 {
            return Err(PyrucastError::Message(format!(
                "decref: refcount already zero for node {}",
                id.0
            )));
        }
        *r -= 1;
        Ok(())
    }

    /// Current refcount of a node (0 for a collected or unknown node).
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// let id = c.add_node(&[0.0, 0.0]).unwrap();
    /// assert_eq!(c.refcount(id), 1);
    /// ```
    pub fn refcount(&self, id: NodeId) -> u32 {
        if self.is_alive(id) {
            self.refcount[id.0 as usize]
        } else {
            0
        }
    }

    /// Garbage collector: mark as collected every live node whose refcount
    /// is 0. Returns the number of collected nodes. Ids are never reused.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// let id = c.add_node(&[0.0, 0.0]).unwrap();
    /// assert_eq!(c.gc(), 0); // un ticket subsiste : rien n'est ramassé
    /// c.decref(id).unwrap();
    /// assert_eq!(c.gc(), 1); // ramassé
    /// ```
    pub fn gc(&mut self) -> usize {
        let mut collected = 0;
        for i in 0..self.alive.len() {
            if self.alive[i] && self.refcount[i] == 0 {
                self.alive[i] = false;
                collected += 1;
            }
        }
        collected
    }

    /// Coordinates of a node in the active configuration. Error if the node was
    /// collected or never existed.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// let id = c.add_node(&[1.5, -2.0]).unwrap();
    /// assert_eq!(c.position(id).unwrap(), &[1.5, -2.0]);
    /// ```
    pub fn position(&self, id: NodeId) -> Result<&[f64]> {
        self.ensure_alive(id)?;
        let d = self.dim as usize;
        let s = id.0 as usize * d;
        Ok(&self.configs[self.active][s..s + d])
    }

    /// Check that **every** id of `ids` names a live node — the whole-list form
    /// of the test [`position`](Self::position) does one node at a time.
    ///
    /// A driver calls this once for a cell connectivity, before its parallel
    /// region, and then reads positions with
    /// `position_alive`: the same guarantee, paid once
    /// per zone instead of once per node of every cell of every call.
    ///
    /// ```
    /// # use pyrucast::atoms::Node;
    /// # use pyrucast::coords::Coords;
    /// # use pyrucast::handle::Handle;
    /// let coords = Handle::new(Coords::new(2)?);
    /// let a = Node::create_in(coords.clone(), &[0.0, 1.0])?;
    /// let b = Node::create_in(coords.clone(), &[2.0, 3.0])?;
    /// coords.read().ensure_all_alive(&[a.id(), b.id()])?;
    /// // Un nœud collecté est refusé — et il l'est **avant** la boucle.
    /// drop(a);
    /// coords.write().gc();
    /// assert!(coords.read().ensure_all_alive(&[b.id()]).is_ok());
    /// # Ok::<(), pyrucast::PyrucastError>(())
    /// ```
    pub fn ensure_all_alive(&self, ids: &[NodeId]) -> Result<()> {
        for &id in ids {
            self.ensure_alive(id)?;
        }
        Ok(())
    }

    /// The position of a node **already known to be live**.
    ///
    /// The unchecked twin of [`position`](Self::position), for a caller that
    /// validated its whole node list with
    /// [`ensure_all_alive`](Self::ensure_all_alive). Reading a dead or unknown
    /// id here panics on the slice bounds rather than returning an error: the
    /// contract is the caller's, and it is a cheap one to honour — one pass over
    /// a connectivity, once per zone.
    pub(crate) fn position_alive(&self, id: NodeId) -> &[f64] {
        let d = self.dim as usize;
        let s = id.0 as usize * d;
        &self.configs[self.active][s..s + d]
    }

    /// Set the coordinates of a node in the active configuration.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// let id = c.add_node(&[0.0, 0.0]).unwrap();
    /// c.set_position(id, &[3.0, 4.0]).unwrap();
    /// assert_eq!(c.position(id).unwrap(), &[3.0, 4.0]);
    /// ```
    pub fn set_position(&mut self, id: NodeId, coords: &[f64]) -> Result<()> {
        self.ensure_alive(id)?;
        if coords.len() != self.dim as usize {
            return Err(PyrucastError::Message(format!(
                "set_position: expected {} coordinates, got {}",
                self.dim,
                coords.len()
            )));
        }
        self.check_radius("set_position", coords)?;
        let d = self.dim as usize;
        let s = id.0 as usize * d;
        self.configs[self.active][s..s + d].copy_from_slice(coords);
        Ok(())
    }

    fn ensure_alive(&self, id: NodeId) -> Result<()> {
        if !self.is_alive(id) {
            return Err(PyrucastError::Message(format!(
                "node {} not found or collected",
                id.0
            )));
        }
        Ok(())
    }

    /// Iterate over live NodeIds in ascending id order.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// let a = c.add_node(&[0.0, 0.0]).unwrap();
    /// let b = c.add_node(&[1.0, 0.0]).unwrap();
    /// c.decref(a).unwrap();
    /// c.gc();
    /// assert_eq!(c.iter_live().collect::<Vec<_>>(), vec![b]); // le mort est sauté
    /// ```
    pub fn iter_live(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.alive
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| a.then_some(NodeId(i as u32)))
    }

    // ─── Configurations ───

    /// Add a new configuration by cloning the active one. Returns its index.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// let deformee = c.add_config("deformed"); // clone de la configuration active
    /// assert_eq!(c.names(), &["default".to_string(), "deformed".to_string()]);
    /// assert_eq!(deformee, 1);
    /// ```
    pub fn add_config(&mut self, name: impl Into<String>) -> usize {
        let copy = self.configs[self.active].clone();
        self.configs.push(copy);
        self.config_names.push(name.into());
        self.configs.len() - 1
    }

    /// Select the active configuration by index.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// let id = c.add_node(&[0.0, 0.0]).unwrap();
    /// let deformee = c.add_config("deformed");
    /// c.select(deformee).unwrap();
    /// c.set_position(id, &[0.1, 0.0]).unwrap(); // n'écrit que dans "deformed"
    /// c.select(0).unwrap();
    /// assert_eq!(c.position(id).unwrap(), &[0.0, 0.0]);
    /// ```
    pub fn select(&mut self, config: usize) -> Result<()> {
        if config >= self.configs.len() {
            return Err(PyrucastError::Message(format!(
                "select: index {} ≥ configuration count ({})",
                config,
                self.configs.len()
            )));
        }
        self.active = config;
        Ok(())
    }

    /// Index of the active configuration.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// assert_eq!(c.active(), 0);
    /// let i = c.add_config("deformed");
    /// c.select(i).unwrap();
    /// assert_eq!(c.active(), i);
    /// ```
    pub fn active(&self) -> usize {
        self.active
    }

    /// Names of the configurations, in order.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let c = Coords::new(2).unwrap();
    /// assert_eq!(c.names(), &["default".to_string()]);
    /// ```
    pub fn names(&self) -> &[String] {
        &self.config_names
    }

    // ─── Solver permutation ───

    /// Current permutation (length = capacity), or `None` for identity.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// assert!(c.permutation().is_none()); // None = identité
    /// for _ in 0..3 { c.add_node(&[0.0, 0.0]).unwrap(); }
    /// c.set_permutation(vec![2, 0, 1]).unwrap();
    /// assert_eq!(c.permutation(), Some(&[2, 0, 1][..]));
    /// ```
    pub fn permutation(&self) -> Option<&[u32]> {
        self.permutation.as_deref()
    }

    /// Set the solver permutation. Its length must equal `capacity`; each
    /// value must be unique and within `[0, capacity)`.
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// for _ in 0..3 { c.add_node(&[0.0, 0.0]).unwrap(); }
    /// // permutation[0] = 2 : le nœud d'id 0 est en position solveur 2.
    /// c.set_permutation(vec![2, 0, 1]).unwrap();
    /// // Ce doit être une vraie permutation des positions.
    /// assert!(c.set_permutation(vec![0, 0, 1]).is_err());
    /// ```
    pub fn set_permutation(&mut self, perm: Vec<u32>) -> Result<()> {
        let cap = self.capacity();
        if perm.len() != cap {
            return Err(PyrucastError::Message(format!(
                "set_permutation: length {} ≠ capacity {}",
                perm.len(),
                cap
            )));
        }
        let cap_u = cap as u32;
        let mut seen = vec![false; cap];
        for &v in &perm {
            if v >= cap_u {
                return Err(PyrucastError::Message(format!(
                    "set_permutation: value {} ≥ capacity {}",
                    v, cap_u
                )));
            }
            let i = v as usize;
            if seen[i] {
                return Err(PyrucastError::Message(format!(
                    "set_permutation: duplicate value {}",
                    v
                )));
            }
            seen[i] = true;
        }
        self.permutation = Some(perm);
        Ok(())
    }

    /// Clear the permutation (back to identity).
    ///
    /// ```
    /// # use pyrucast::coords::Coords;
    /// let mut c = Coords::new(2).unwrap();
    /// for _ in 0..3 { c.add_node(&[0.0, 0.0]).unwrap(); }
    /// c.set_permutation(vec![2, 0, 1]).unwrap();
    /// c.clear_permutation();
    /// assert!(c.permutation().is_none());
    /// ```
    pub fn clear_permutation(&mut self) {
        self.permutation = None;
    }
}

impl fmt::Debug for Coords {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Coords")
            .field("dim", &self.dim)
            .field("frame", &self.frame)
            .field("configs", &self.config_names)
            .field("active", &self.active)
            .field("node_count", &self.node_count())
            .field("capacity", &self.capacity())
            .field("permutation", &self.permutation.is_some())
            .finish()
    }
}

impl fmt::Display for Coords {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let active_name = &self.config_names[self.active];
        let collected = self.capacity() - self.node_count();
        let perm_label = if self.permutation.is_some() {
            "custom"
        } else {
            "identity"
        };
        // The frame is shown only when it departs from the Cartesian default, so
        // the overwhelmingly common rendering stays unchanged.
        let frame_label = match self.frame {
            CoordinateFrame::Cartesian => String::new(),
            f => format!(" ({f})"),
        };
        write!(
            f,
            "Coords: dim={}{}, configs={} (active=\"{}\"), nodes={} ({} collected), permutation: {}",
            self.dim,
            frame_label,
            self.configs.len(),
            active_name,
            self.node_count(),
            collected,
            perm_label
        )
    }
}

impl crate::dump::Dump for Coords {
    fn render(&self, opts: &crate::dump::DumpOptions) -> String {
        use crate::dump::{fmt_float, table};
        let dim = self.dim as usize;
        const AXES: [&str; 3] = ["x", "y", "z"];
        // In a meridian plane the columns are the radius and the axis, so name
        // them as such rather than x/y.
        const AXES_AXI: [&str; 2] = ["r", "z"];
        let axes: &[&str] = if self.frame.is_axisymmetric() {
            &AXES_AXI
        } else {
            &AXES
        };
        let mut headers = vec!["node".to_string()];
        headers.extend((0..dim).map(|i| axes.get(i).copied().unwrap_or("?").to_string()));
        headers.push("refs".to_string());
        let rows: Vec<Vec<String>> = self
            .iter_live()
            .map(|id| {
                let mut row = vec![id.to_string()];
                match self.position(id) {
                    Ok(c) => row.extend(c.iter().map(|v| fmt_float(*v, opts.precision))),
                    Err(_) => row.extend((0..dim).map(|_| "?".to_string())),
                }
                row.push(self.refcount(id).to_string());
                row
            })
            .collect();
        format!("{self}\n{}", table(&headers, &rows, opts))
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

// ─── Archive ────────────────────────────────────────────────────────────────

impl crate::archive::Archivable for Coords {
    const TAG: &'static str = "Coords";

    /// The per-node refcount is not in the file (see the field's own note).
    /// Size it to the node capacity, at zero: each reloaded `SubMesh` will
    /// re-increment what it references, and the post-order guarantees they come
    /// after this.
    fn on_load(&mut self) {
        self.refcount = vec![0; self.alive.len()];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_dim() {
        let c = Coords::new(3).unwrap();
        assert_eq!(c.dim(), 3);
        assert_eq!(c.node_count(), 0);
        assert_eq!(c.capacity(), 0);
    }

    #[test]
    fn cartesian_by_default_axisymmetric_on_demand() {
        assert!(!Coords::new(2).unwrap().is_axisymmetric());
        let c = Coords::axisymmetric().unwrap();
        assert_eq!(c.dim(), 2);
        assert_eq!(c.frame(), CoordinateFrame::Axisymmetric);
        assert!(c.is_axisymmetric());
        // The frame surfaces in Display only when it is not the default.
        assert!(Coords::new(2).unwrap().to_string().contains("dim=2,"));
        assert!(c.to_string().contains("dim=2 (axisymmetric)"));
    }

    /// `x = r` is a radius: negative values are a modelling error, caught at the
    /// door rather than as a negative `|J|` deep in an integral.
    #[test]
    fn axisymmetric_rejects_a_negative_radius() {
        let mut c = Coords::axisymmetric().unwrap();
        let err = c.add_node(&[-1.0, 0.0]).unwrap_err();
        assert!(format!("{err}").contains("negative radius"));
        // On the axis (r = 0) is legitimate.
        let id = c.add_node(&[0.0, 2.0]).unwrap();
        assert!(c.set_position(id, &[-0.5, 2.0]).is_err());
        c.set_position(id, &[0.5, 2.0]).unwrap();
        // A Cartesian Coords keeps accepting negative x.
        assert!(Coords::new(2).unwrap().add_node(&[-1.0, 0.0]).is_ok());
    }

    #[test]
    fn dim_zero_rejected() {
        let err = Coords::new(0).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn add_node_initializes_refcount_to_one() {
        let mut c = Coords::new(2).unwrap();
        let a = c.add_node(&[0.0, 0.0]).unwrap();
        assert_eq!(c.refcount(a), 1);
        assert!(c.is_alive(a));
    }

    #[test]
    fn add_node_invalid_dim() {
        let mut c = Coords::new(3).unwrap();
        let err = c.add_node(&[1.0, 2.0]).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn gc_does_not_collect_referenced_nodes() {
        let mut c = Coords::new(2).unwrap();
        let a = c.add_node(&[0.0, 0.0]).unwrap();
        // refcount = 1, gc must not collect anything
        assert_eq!(c.gc(), 0);
        assert!(c.is_alive(a));
        // After decref, refcount drops to 0
        c.decref(a).unwrap();
        assert_eq!(c.refcount(a), 0);
        // but only gc actually removes
        assert_eq!(c.gc(), 1);
        assert!(!c.is_alive(a));
    }

    #[test]
    fn incref_protects_from_gc() {
        let mut c = Coords::new(1).unwrap();
        let a = c.add_node(&[3.0]).unwrap();
        c.incref(a).unwrap(); // refcount = 2
        c.decref(a).unwrap(); // refcount = 1
        assert_eq!(c.gc(), 0);
        c.decref(a).unwrap(); // refcount = 0
        assert_eq!(c.gc(), 1);
    }

    #[test]
    fn id_not_reused_after_gc() {
        let mut c = Coords::new(1).unwrap();
        let a = c.add_node(&[0.0]).unwrap();
        c.decref(a).unwrap();
        c.gc();
        let b = c.add_node(&[1.0]).unwrap();
        assert_ne!(a.0, b.0);
        assert_eq!(b.0, 1);
        assert_eq!(c.capacity(), 2);
    }

    #[test]
    fn coord_after_gc_is_error() {
        let mut c = Coords::new(1).unwrap();
        let a = c.add_node(&[42.0]).unwrap();
        c.decref(a).unwrap();
        c.gc();
        assert!(c.position(a).is_err());
        assert!(c.set_position(a, &[0.0]).is_err());
        assert!(c.incref(a).is_err());
        assert!(c.decref(a).is_err());
    }

    #[test]
    fn decref_at_zero_is_error() {
        let mut c = Coords::new(1).unwrap();
        let a = c.add_node(&[0.0]).unwrap();
        c.decref(a).unwrap();
        let err = c.decref(a).unwrap_err();
        assert!(matches!(err, PyrucastError::Message(_)));
    }

    #[test]
    fn set_position_modifies_active_config() {
        let mut c = Coords::new(2).unwrap();
        let a = c.add_node(&[0.0, 0.0]).unwrap();
        c.set_position(a, &[3.0, 4.0]).unwrap();
        assert_eq!(c.position(a).unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn multiple_configs_and_select() {
        let mut c = Coords::new(2).unwrap();
        let a = c.add_node(&[0.0, 0.0]).unwrap();
        let s2 = c.add_config("deformed");
        assert_eq!(s2, 1);
        c.select(s2).unwrap();
        c.set_position(a, &[10.0, 20.0]).unwrap();
        c.select(0).unwrap();
        assert_eq!(c.position(a).unwrap(), &[0.0, 0.0]);
        c.select(1).unwrap();
        assert_eq!(c.position(a).unwrap(), &[10.0, 20.0]);
        assert_eq!(c.names(), &["default".to_string(), "deformed".to_string()]);
    }

    #[test]
    fn select_invalid() {
        let mut c = Coords::new(2).unwrap();
        assert!(c.select(5).is_err());
    }

    #[test]
    fn iter_live_skips_collected() {
        let mut c = Coords::new(1).unwrap();
        let a = c.add_node(&[0.0]).unwrap();
        let b = c.add_node(&[1.0]).unwrap();
        let _cc = c.add_node(&[2.0]).unwrap();
        c.decref(b).unwrap();
        c.gc();
        let live: Vec<u32> = c.iter_live().map(|n| n.0).collect();
        assert_eq!(live, vec![a.0, 2]);
    }

    #[test]
    fn permutation_validation_and_invariant() {
        let mut c = Coords::new(1).unwrap();
        for k in 0..4 {
            c.add_node(&[k as f64]).unwrap();
        }
        c.set_permutation(vec![3, 2, 1, 0]).unwrap();
        assert_eq!(c.permutation(), Some(&[3u32, 2, 1, 0][..]));
        assert!(c.set_permutation(vec![0, 0, 1, 2]).is_err());
        assert!(c.set_permutation(vec![0, 1, 2, 99]).is_err());
        assert!(c.set_permutation(vec![0, 1, 2]).is_err());
        c.clear_permutation();
        assert!(c.permutation().is_none());
    }

    #[test]
    fn permutation_extended_by_add_node() {
        let mut c = Coords::new(1).unwrap();
        for k in 0..3 {
            c.add_node(&[k as f64]).unwrap();
        }
        c.set_permutation(vec![2, 1, 0]).unwrap();
        c.add_node(&[42.0]).unwrap();
        let perm = c.permutation().unwrap();
        assert_eq!(perm.len(), 4);
        let mut sorted = perm.to_vec();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn debug_display() {
        let mut c = Coords::new(2).unwrap();
        c.add_node(&[0.0, 0.0]).unwrap();
        c.add_node(&[1.0, 1.0]).unwrap();
        let d = format!("{:?}", c);
        assert!(d.contains("Coords"));
        assert!(d.contains("dim"));
        let s = format!("{}", c);
        assert!(s.contains("dim=2"));
        assert!(s.contains("nodes=2"));
        assert!(s.contains("identity"));
    }

    #[test]
    fn nodeid_display_and_debug() {
        let n = NodeId(7);
        assert_eq!(format!("{}", n), "7");
        assert_eq!(format!("{:?}", n), "NodeId(7)");
    }
}
