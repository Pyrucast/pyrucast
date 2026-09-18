//! The three method shapes the field wrappers repeat often enough to earn a
//! macro.
//!
//! A macro only pays for itself through its **number of expansions**: the
//! transform method is generated eleven times per family, the binary slot eight
//! times, the component mutator four. Rarer shapes — `min`, `sum`,
//! `components`, `__pow__`, `__richcmp__`… — are written by hand in
//! `node_field.rs` and `element_field.rs`, where they read in one go.
//!
//! Every shape comes as two macros, one per field family: `impl_field_*` for the
//! aggregates (`PyNodeField`, `PyElementField`), which hold their value in
//! `self.inner`; `impl_subfield_*` for the sub-containers (`PySubNodeField`,
//! `PySubElementField`), which read it through `self.handle`. Two macros rather
//! than one family parameter: each body then names its trait and its access
//! outright, and reads without a detour.
//!
//! A macro's name states the **item it produces**, not what is handed to it: it
//! binds any function of the expected shape, and today's inventory is no
//! property of the macro.

/// Puts the same block of methods on every type of the list, in one
/// `#[pymethods]` per type. This is the only code the macros of this file share.
///
/// Arguments: `stub` or `bare`, depending on whether the block must be decorated
/// for pyo3-stub-gen; the list of types; then the block itself, braces included.
///
/// ```ignore
/// impl_pymethods_for_each!(stub [PyNodeField, PyElementField] {
///     fn zero(&self) -> f64 { 0.0 }
/// });
/// ```
///
/// **The block arrives as a single `tt`**, not taken apart. That is what allows
/// the documentation to be written as `///` at the call site: captured line by
/// line (`$(#[doc = $doc:literal])*`), it cannot repeat inside a loop over the
/// types — rustc refuses two repetitions of different lengths at one level. A
/// `tt`, on the other hand, is copied freely.
///
/// **The block names its type `Self`**, since it does not know which one will
/// carry it.
///
/// `bare` is no longer used by this file: it remains for a block that must not be
/// declared to the stub, as `__richcmp__` is — CPython exposes it under the four
/// names `__ge__`/`__gt__`/`__le__`/`__lt__`.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_pymethods_for_each {
    (stub [$($T:ident),+ $(,)?] $corps:tt) => {
        $(
            #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
            #[::pyo3::pymethods]
            impl $T $corps
        )+
    };
    (bare [$($T:ident),+ $(,)?] $corps:tt) => {
        $(
            #[::pyo3::pymethods]
            impl $T $corps
        )+
    };
}

/// Generates, on an **aggregate**, an argument-less method returning a field of
/// the same flavour: it applies to the whole field the `ops::field` function
/// bearing its name, which takes the field alone.
///
/// Arguments: the documentation as `///`, the list of types, the method name —
/// which is also the name of the `ops::field` function called.
///
/// ```ignore
/// impl_field_transform_pymethod! {
///     /// Element-wise square root of a field (`nan` for negatives).
///     [PyNodeField, PyElementField], sqrt
/// }
/// ```
///
/// generates, on each of the two types:
///
/// ```ignore
/// fn sqrt(&self) -> PyResult<Self> {
///     let out = ops::field::sqrt(&self.inner)?;
///     Ok(Self { inner: out })
/// }
/// ```
///
/// Today's eleven calls — `sqrt`, `exp`, `cos`… — go through
/// `define_polymorphic_pyfunction!` (`py/ops/field.rs`), which generates in one
/// go the dispatching free function and the methods of all four flavours, from a
/// text written once.
#[macro_export]
macro_rules! impl_field_transform_pymethod {
    ($(#[doc = $doc:literal])* [$($T:ident),+ $(,)?], $nom:ident) => {
        $crate::impl_pymethods_for_each!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self) -> ::pyo3::PyResult<Self> {
                let out = $crate::ops::field::$nom(&self.inner)?;
                Ok(Self { inner: out })
            }
        });
    };
}

/// The same, on a **sub-container**: the value is read through the handle, and
/// the result gets a fresh one.
#[macro_export]
macro_rules! impl_subfield_transform_pymethod {
    ($(#[doc = $doc:literal])* [$($T:ident),+ $(,)?], $nom:ident) => {
        $crate::impl_pymethods_for_each!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self) -> ::pyo3::PyResult<Self> {
                let out = $crate::ops::field::$nom(&*self.handle.read())?;
                Ok(Self {
                    handle: $crate::handle::Handle::new(out),
                })
            }
        });
    };
}

/// Generates, on an **aggregate**, a binary operator slot (`__add__`, `__sub__`,
/// `__mul__`, `__truediv__`): it hands the right-hand side and the closure to the
/// `binary` dispatcher each flavour defines in its inherent block.
///
/// Arguments: the documentation as `///`, the list of types, the slot name, and
/// the closure applied term by term.
///
/// ```ignore
/// impl_field_binary_pyslot! {
///     /// `field + other` — element-wise sum.
///     [PyNodeField], __add__, |a, b| a + b
/// }
/// ```
///
/// **To be called from the module declaring the `#[pyclass]`**, with a
/// single-type list: pyo3 generates for a slot an `unsafe fn` trampoline calling
/// another one, and edition 2024 no longer covers that body implicitly —
/// `unsafe_op_in_unsafe_fn` fires as soon as the `impl` lives elsewhere.
#[macro_export]
macro_rules! impl_field_binary_pyslot {
    ($(#[doc = $doc:literal])* [$($T:ident),+ $(,)?], $nom:ident, $f:expr) => {
        $crate::impl_pymethods_for_each!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self, rhs: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<Self> {
                self.binary(rhs, $f)
            }
        });
    };
}

/// The same, on a **sub-container**: its dispatcher is called
/// `scalar_or_combine` — a scalar applies everywhere, another sub-field combines
/// element by element.
#[macro_export]
macro_rules! impl_subfield_binary_pyslot {
    ($(#[doc = $doc:literal])* [$($T:ident),+ $(,)?], $nom:ident, $f:expr) => {
        $crate::impl_pymethods_for_each!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self, rhs: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<Self> {
                self.scalar_or_combine(rhs, $f)
            }
        });
    };
}

/// Generates, on an **aggregate**, a component mutation by a scalar
/// (`add_to_component` and its three siblings). The mutation reaches down to the
/// zones defining the component, and the method returns nothing.
///
/// Arguments: the documentation as `///`, the list of types, the method name —
/// which is also the name of the `Field` trait method called.
///
/// ```ignore
/// impl_field_mutator_pymethod! {
///     /// Add `scalar` to `component` on every zone that defines it.
///     [PyNodeField, PyElementField], add_to_component
/// }
/// ```
#[macro_export]
macro_rules! impl_field_mutator_pymethod {
    ($(#[doc = $doc:literal])* [$($T:ident),+ $(,)?], $nom:ident) => {
        $crate::impl_pymethods_for_each!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self, component: &str, scalar: f64) -> ::pyo3::PyResult<()> {
                use $crate::containers::field::Field;
                self.inner.$nom(component, scalar)?;
                Ok(())
            }
        });
    };
}

/// The same, on a **sub-container**: the mutation happens in place, in its own
/// zone, under the only **write** guard these macros take.
#[macro_export]
macro_rules! impl_subfield_mutator_pymethod {
    ($(#[doc = $doc:literal])* [$($T:ident),+ $(,)?], $nom:ident) => {
        $crate::impl_pymethods_for_each!(stub [$($T),+] {
            $(#[doc = $doc])*
            fn $nom(&self, component: &str, scalar: f64) -> ::pyo3::PyResult<()> {
                use $crate::containers::field::SubField;
                self.handle.write().$nom(component, scalar)?;
                Ok(())
            }
        });
    };
}
