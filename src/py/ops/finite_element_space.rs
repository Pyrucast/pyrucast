//! Python wrappers for [`crate::ops::finite_element_space`] — the operators
//! producing a finite-element space from another.

use crate::py::finite_element_space::PyFiniteElementSpace;
use crate::py::mesh::PyMesh;
use pyo3::prelude::*;
use pyrucast_macros::py_op;

/// Restrict `fespace` to the cells of `mesh`, each subspace keeping its
/// interpolation and its quadrature.
///
/// A cell is kept when `mesh` holds it too — same type, same nodes in the same
/// order, as `pyrucast.mesh.intersection`. It is the way to a field at the
/// Gauss points on part of a mesh:
/// `pyrucast.element_field.restrict(f, f.support().restrict(part))`.
///
/// # Parameters
///
/// - `fespace`: the space to cut down.
/// - `mesh`: the cells to keep.
///
/// # Returns
///
/// A `FiniteElementSpace` mirroring `fespace` zone by zone, possibly with
/// empty subspaces. A subspace whose zone is kept whole comes back as it is,
/// shared, so a field laid on it still combines with one laid on the result.
#[py_op(method_on = PyFiniteElementSpace)]
#[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
#[pyfunction]
#[pyo3(name = "restrict_fespace")]
pub fn restrict(fespace: PyRef<PyFiniteElementSpace>, mesh: PyRef<PyMesh>) -> PyFiniteElementSpace {
    PyFiniteElementSpace {
        inner: crate::ops::finite_element_space::restrict(&fespace.inner, &mesh.inner),
    }
}
