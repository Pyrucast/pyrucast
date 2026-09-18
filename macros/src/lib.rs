//! pyrucast's procedural macros.
//!
//! Only one: [`py_op`], which derives from a free function exposed to Python
//! the **method** of its subject (`CONVENTIONS.md`, § "Le verbe exposé aussi
//! en méthode").
//!
//! Why a procedural macro rather than a `macro_rules!`: one must *read* an
//! already written signature to strip its first parameter, and turn
//! `#[pyo3(signature = (mesh, angle_deg=None))]` into the same list minus its
//! first entry. A `macro_rules!` sees only the tokens handed to it and cannot
//! open a captured fragment; a procedural macro receives the whole item and
//! takes it apart.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::{parse_macro_input, FnArg, ItemFn, LitStr, Meta};

/// Exposes a free `#[pyfunction]` **also** as a method of its first argument,
/// without rewriting either its signature or its documentation.
///
/// The attribute goes **above** all the others: attribute macros apply from
/// the outside in, and this one must still see `#[pyfunction]` and
/// `#[pyo3(signature = …)]` attached to the item.
///
/// ```ignore
/// #[py_op(method_on = PyMesh)]
/// #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
/// #[pyfunction]
/// #[pyo3(signature = (mesh, angle_deg=None))]
/// pub fn skin(mesh: PyRef<PyMesh>, angle_deg: Option<f64>) -> PyResult<PyMesh> {
///     todo!()
/// }
/// ```
///
/// The function is re-emitted **unchanged**, followed by:
///
/// ```ignore
/// #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
/// #[pymethods]
/// impl PyMesh {
///     // …the function's documentation, copied over as literals…
///     #[pyo3(signature = (angle_deg=None))]
///     fn skin(slf: PyRef<'_, Self>, angle_deg: Option<f64>) -> PyResult<PyMesh> {
///         self::skin(slf, angle_deg)
///     }
/// }
/// ```
///
/// Copying the documentation, rather than pointing at it, is the whole point:
/// it thereby reaches `__doc__` **and** the `.pyi` stub the IDEs read, where
/// a "See …" pointer would stay dead text.
///
/// Parameters: `method_on = PyType` (required), and `name = "other_name"` when
/// the method does not bear the function's name — the qualifier the module
/// gave the free function must sometimes move into the method's name
/// (`matrix::stiffness` → `stiffness_matrix`).
#[proc_macro_attribute]
pub fn py_op(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut method_on: Option<syn::Ident> = None;
    let mut rename: Option<LitStr> = None;
    let parser = syn::meta::parser(|meta| {
        if meta.path.is_ident("method_on") {
            method_on = Some(meta.value()?.parse()?);
            Ok(())
        } else if meta.path.is_ident("name") {
            rename = Some(meta.value()?.parse()?);
            Ok(())
        } else {
            Err(meta.error("`py_op` accepts `method_on = PyType` and `name = \"…\"`"))
        }
    });
    parse_macro_input!(attr with parser);

    let func = parse_macro_input!(item as ItemFn);
    let Some(method_on) = method_on else {
        return error("`py_op` requires `method_on = PyType`, the type that will carry the method");
    };

    // The subject is the first parameter; the following ones are copied as they
    // are, names included — precisely what a `macro_rules!` could not do from an
    // already written signature.
    //
    // Except that a `Python` token may precede it: pyo3 supplies it to the
    // function without it appearing in the Python signature. The subject is then
    // the second parameter, and the call will have to put the `py` back in front.
    let mut inputs = func.sig.inputs.iter();
    let Some(first) = inputs.next() else {
        return error("`py_op` on a function without arguments: there is no subject");
    };
    let (py_param, subject) = if type_mentions(first, "Python") {
        match inputs.next() {
            Some(subject) => (Some(first), subject),
            None => return error("`py_op`: after the `py: Python`, the subject is missing"),
        }
    } else {
        (None, first)
    };
    if type_mentions(subject, "Bound") {
        return error(
            "`py_op` does not suit a polymorphic subject: the method would return `Any` \
             where its receiver fixes the type. Extract one function per flavour, with a \
             `PyRef<…>` subject, and place the attribute on each of them (model: \
             `py::ops::node_field::mask`)",
        );
    }
    let rest: Vec<&FnArg> = inputs.collect();
    let forwarded: Vec<_> = rest
        .iter()
        .filter_map(|arg| match arg {
            FnArg::Typed(pat) => Some(&pat.pat),
            FnArg::Receiver(_) => None,
        })
        .collect();

    // Order of the method's parameters: the receiver, then the `py` if there is
    // one, then the rest — the shape the hand-written methods impose. The call
    // itself restores the free function's order.
    let py_pat = py_param.and_then(|arg| match arg {
        FnArg::Typed(pat) => Some(&pat.pat),
        FnArg::Receiver(_) => None,
    });
    let mut method_args: Vec<TokenStream2> = Vec::new();
    if let Some(py) = py_param {
        method_args.push(quote! { #py });
    }
    method_args.extend(rest.iter().map(|arg| quote! { #arg }));
    let mut call_args: Vec<TokenStream2> = Vec::new();
    if let Some(py) = py_pat {
        call_args.push(quote! { #py });
    }
    call_args.push(quote! { slf });
    call_args.extend(forwarded.iter().map(|pat| quote! { #pat }));

    // The method inherits what the function carries: its documentation, and its
    // lint waivers. `solver::solve_unilateral` proves it by example — nine
    // arguments, an `#[allow(clippy::too_many_arguments)]` assumed on the
    // function, and a method with just as many, the receiver replacing the
    // subject. Without this copying, the waiver stopped at the function and the
    // lint struck code nobody had written.
    //
    //
    // `allow` only, never `expect`: an `expect` whose lint does not fire on the
    // method would itself become a warning.
    let herites: Vec<_> = func
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc") || attr.path().is_ident("allow"))
        .collect();
    let signature = signature_without_subject(&func);
    let output = &func.sig.output;
    let free = &func.sig.ident;
    let method = match &rename {
        Some(lit) => syn::Ident::new(&lit.value(), lit.span()),
        None => free.clone(),
    };

    quote! {
        #func

        #[cfg_attr(feature = "stub-gen", ::pyo3_stub_gen::derive::gen_stub_pymethods)]
        #[::pyo3::pymethods]
        impl #method_on {
            #(#herites)*
            #signature
            fn #method(slf: ::pyo3::PyRef<'_, Self>, #(#method_args),*) #output {
                self::#free(#(#call_args),*)
            }
        }
    }
    .into()
}

/// The function's `#[pyo3(signature = (…))]` attribute, minus its first entry
/// — the subject, which the method receives through its receiver. `None` when
/// the function carries none: all its arguments are then required, and the
/// method has nothing to declare either.
fn signature_without_subject(func: &ItemFn) -> Option<TokenStream2> {
    let list = func.attrs.iter().find_map(|attr| {
        if !attr.path().is_ident("pyo3") {
            return None;
        }
        let Meta::List(list) = &attr.meta else {
            return None;
        };
        let mut tokens = list.tokens.clone().into_iter();
        match (tokens.next(), tokens.next(), tokens.next()) {
            (
                Some(TokenTree::Ident(key)),
                Some(TokenTree::Punct(eq)),
                Some(TokenTree::Group(g)),
            ) if key == "signature" && eq.as_char() == '=' => Some(g.stream()),
            _ => None,
        }
    })?;
    let rest = drop_first_entry(list);
    Some(quote! { #[pyo3(signature = (#rest))] })
}

/// Everything after the first top-level comma. The entries of a pyo3
/// signature are heterogeneous (`mesh`, `angle_deg=None`, `*`, `**kwargs`),
/// so we cut on the comma rather than try to parse them.
fn drop_first_entry(tokens: TokenStream2) -> TokenStream2 {
    let mut iter = tokens.into_iter().skip_while(|tree| !is_comma(tree));
    iter.next(); // the comma itself
    iter.collect()
}

fn is_comma(tree: &TokenTree) -> bool {
    matches!(tree, TokenTree::Punct(p) if p.as_char() == ',')
}

/// Does this parameter's type name `ident`? Read by tokens: the type is
/// written `Python<'_>`, `PyRef<PyMesh>`, `&Bound<'_, PyAny>`…
fn type_mentions(arg: &FnArg, ident: &str) -> bool {
    let FnArg::Typed(pat) = arg else {
        return false;
    };
    quote! { #pat }.to_string().contains(ident)
}

fn error(message: &str) -> TokenStream {
    syn::Error::new(Span::call_site(), message)
        .to_compile_error()
        .into()
}

#[cfg(test)]
mod tests {
    use super::drop_first_entry;
    use quote::quote;

    #[test]
    fn le_sujet_disparait_le_reste_est_intact() {
        let reste = drop_first_entry(quote! { mesh, angle_deg = None });
        assert_eq!(reste.to_string(), quote! { angle_deg = None }.to_string());
    }

    /// A function whose subject is its only argument (`consolidate`, `orient`…):
    /// the method then has nothing left to declare.
    #[test]
    fn un_sujet_seul_ne_laisse_rien() {
        assert!(drop_first_entry(quote! { mesh }).is_empty());
    }

    /// The case this test exists for: a **nested** comma lives inside a group,
    /// hence outside the level where the separator is looked for. Cutting on it
    /// would sever the signature at the wrong place — and the code would still
    /// compile, shipping a wrong Python API.
    #[test]
    fn une_virgule_imbriquee_ne_coupe_pas() {
        let reste = drop_first_entry(quote! { mesh, kinds = (1, 2), tol = None });
        assert_eq!(
            reste.to_string(),
            quote! { kinds = (1, 2), tol = None }.to_string()
        );
    }
}
