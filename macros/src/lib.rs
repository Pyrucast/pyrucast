//! Macros procédurales de pyrucast.
//!
//! Une seule : [`py_op`], qui dérive d'une fonction libre exposée à Python la
//! **méthode** de son sujet (`CONVENTIONS.md`, § « Le verbe exposé aussi en
//! méthode »).
//!
//! Pourquoi une macro procédurale et pas une `macro_rules!` : il faut *lire*
//! une signature déjà écrite pour en retirer le premier paramètre, et
//! transformer `#[pyo3(signature = (mesh, angle_deg=None))]` en la même liste
//! privée de sa première entrée. Une `macro_rules!` ne voit que les jetons
//! qu'on lui passe et ne sait pas ouvrir un fragment capturé ; une macro
//! procédurale reçoit l'item entier et le décompose.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::{parse_macro_input, FnArg, ItemFn, LitStr, Meta};

/// Expose une fonction libre `#[pyfunction]` **aussi** comme méthode de son
/// premier argument, sans réécrire ni sa signature ni sa documentation.
///
/// L'attribut se pose **au-dessus** de tous les autres : les macros d'attribut
/// s'appliquent de l'extérieur vers l'intérieur, et celle-ci doit encore voir
/// `#[pyfunction]` et `#[pyo3(signature = …)]` attachés à l'item.
///
/// ```ignore
/// #[py_op(method_on = PyMesh)]
/// #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pyfunction)]
/// #[pyfunction]
/// #[pyo3(signature = (mesh, angle_deg=None))]
/// pub fn skin(mesh: PyRef<PyMesh>, angle_deg: Option<f64>) -> PyResult<PyMesh> { … }
/// ```
///
/// La fonction est réémise **inchangée**, suivie de :
///
/// ```ignore
/// #[cfg_attr(feature = "stub-gen", pyo3_stub_gen::derive::gen_stub_pymethods)]
/// #[pymethods]
/// impl PyMesh {
///     /// …la documentation de la fonction, recopiée en littéraux…
///     #[pyo3(signature = (angle_deg=None))]
///     fn skin(slf: PyRef<'_, Self>, angle_deg: Option<f64>) -> PyResult<PyMesh> {
///         self::skin(slf, angle_deg)
///     }
/// }
/// ```
///
/// Recopier la documentation, plutôt que d'y renvoyer, est tout l'intérêt :
/// elle atteint ainsi `__doc__` **et** le stub `.pyi` que lisent les IDE, alors
/// qu'un pointeur « Voir … » y resterait du texte mort.
///
/// Paramètres : `method_on = PyType` (obligatoire), et `name = "autre_nom"`
/// quand la méthode ne porte pas le nom de la fonction — le qualificatif que le
/// module donnait à la fonction libre doit parfois passer dans le nom de la
/// méthode (`matrix::stiffness` → `stiffness_matrix`).
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
            Err(meta.error("`py_op` accepte `method_on = PyType` et `name = \"…\"`"))
        }
    });
    parse_macro_input!(attr with parser);

    let func = parse_macro_input!(item as ItemFn);
    let Some(method_on) = method_on else {
        return error("`py_op` exige `method_on = PyType`, le type qui portera la méthode");
    };

    // Le sujet est le premier paramètre ; les suivants sont recopiés tels
    // quels, noms compris — c'est précisément ce qu'une `macro_rules!` ne
    // saurait pas faire d'une signature déjà écrite.
    let mut inputs = func.sig.inputs.iter();
    let Some(subject) = inputs.next() else {
        return error("`py_op` sur une fonction sans argument : il n'y a pas de sujet");
    };
    if type_mentions(subject, "Python") {
        return error(
            "`py_op` ne gère pas encore un `py: Python` en tête : le sujet doit être \
             le premier paramètre",
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

    let docs: Vec<_> = func
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
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
            #(#docs)*
            #signature
            fn #method(slf: ::pyo3::PyRef<'_, Self>, #(#rest),*) #output {
                self::#free(slf, #(#forwarded),*)
            }
        }
    }
    .into()
}

/// L'attribut `#[pyo3(signature = (…))]` de la fonction, privé de sa première
/// entrée — le sujet, que la méthode reçoit par son receveur. `None` quand la
/// fonction n'en porte pas : tous ses arguments sont alors obligatoires, et la
/// méthode n'a rien à déclarer non plus.
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

/// Tout ce qui suit la première virgule de premier niveau. Les entrées d'une
/// signature pyo3 sont hétérogènes (`mesh`, `angle_deg=None`, `*`, `**kwargs`),
/// donc on coupe sur la virgule plutôt que d'essayer de les analyser.
fn drop_first_entry(tokens: TokenStream2) -> TokenStream2 {
    let mut iter = tokens.into_iter().skip_while(|tree| !is_comma(tree));
    iter.next(); // la virgule elle-même
    iter.collect()
}

fn is_comma(tree: &TokenTree) -> bool {
    matches!(tree, TokenTree::Punct(p) if p.as_char() == ',')
}

/// Le type de ce paramètre nomme-t-il `ident` ? Lecture par jetons : le type
/// s'écrit `Python<'_>`, `PyRef<PyMesh>`, `&Bound<'_, PyAny>`…
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

    /// Une fonction dont le sujet est le seul argument (`consolidate`,
    /// `orient`…) : la méthode n'a plus rien à déclarer.
    #[test]
    fn un_sujet_seul_ne_laisse_rien() {
        assert!(drop_first_entry(quote! { mesh }).is_empty());
    }

    /// Le cas qui justifie ce test : une virgule **imbriquée** vit dans un
    /// groupe, donc hors du niveau où l'on cherche le séparateur. Couper
    /// dessus amputerait la signature au mauvais endroit — et le code
    /// compilerait quand même, en livrant une API Python fausse.
    #[test]
    fn une_virgule_imbriquee_ne_coupe_pas() {
        let reste = drop_first_entry(quote! { mesh, kinds = (1, 2), tol = None });
        assert_eq!(
            reste.to_string(),
            quote! { kinds = (1, 2), tol = None }.to_string()
        );
    }
}
