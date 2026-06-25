# Interrompre une fonction

Certains opérateurs (mailleurs, solveurs, boucles de raffinement) peuvent
tourner longtemps. On veut alors pouvoir les **interrompre** proprement —
typiquement par un `Ctrl+C` depuis Python, mais aussi par un *timeout* ou un
signal externe depuis un programme Rust. Ce chapitre explique le principe et
comment l'implémenter pour un nouvel opérateur.

## Pourquoi un `Ctrl+C` ne suffit pas

Quand une fonction Rust appelée depuis Python tourne dans une longue boucle,
elle **garde le GIL et ne rend jamais la main** à l'interpréteur. Or
`KeyboardInterrupt` n'est levée par Python qu'**entre deux bytecodes**. Le
`Ctrl+C` (signal `SIGINT`) est donc bien *enregistré*, mais il ne se
déclenchera qu'au **retour** de la fonction Rust : pendant le calcul, la
combinaison paraît sans effet.

La solution est l'**interruption coopérative** : la boucle longue vérifie
elle-même, à intervalles réguliers, s'il faut s'arrêter.

## Le principe : un jeton, pas un détail de frontend

L'écueil serait de faire appeler `Python::check_signals` directement par
l'opérateur — cela couplerait le **cœur de calcul** à PyO3 et casserait
l'usage en Rust pur (cf. [Compilation et tests](../compilation.md), build sans
`python-api`).

À la place, l'opérateur reçoit un **jeton d'interruption** abstrait — le trait
`Cancel` du module `interrupt` — qu'il interroge périodiquement. C'est le
**frontend** qui décide *comment* l'interruption est signalée :

| Frontend | Jeton | Déclencheur |
|---|---|---|
| Rust pur | `NoCancel` / `()` | jamais |
| Rust pur | `AtomicBool` | un autre thread / handler `ctrlc` lève le drapeau |
| Rust pur | `Deadline` | dépassement d'un délai (timeout) |
| Python | `PySignals` (dans `src/py`, gaté `python-api`) | `Ctrl+C` via `Python::check_signals` |

Le trait est du **Rust pur, sans PyO3** :

```rust,ignore
pub trait Cancel {
    /// `Ok(())` pour continuer, `Err(Interrupted)` pour s'arrêter proprement.
    fn check(&self) -> Result<()>;
}
```

L'erreur renvoyée est `PyrucastError::Interrupted`. Sa conversion vers Python
(gatée `python-api`) produit une vraie **`KeyboardInterrupt`**, pas un
`RuntimeError` générique. Le cœur, lui, ne voit que `&dyn Cancel` : il reste
PyO3-free et utilisable depuis un programme Rust.

## Implémenter l'interruption dans un opérateur

Trois gestes.

**1. Faire passer le jeton et le sonder.** Le cœur de calcul prend un
`&dyn Cancel` et l'interroge à chaque tour de boucle (un tour = un événement
*grossier* — un élément, une couche, une itération de solveur — pour que le
coût d'un `check` soit négligeable) :

```rust,ignore
pub fn pave(/* … */, cancel: &dyn Cancel) -> Result<…> {
    loop {
        cancel.check()?;          // ← point d'interruption
        // … une étape de travail …
    }
}
```

> Sonder une fois par étape grossière suffit ; inutile de throttler par un
> compteur si chaque tour fait déjà un travail substantiel.

**2. Exposer deux formes côté Rust** — une simple et une interruptible — pour
ne pas imposer un jeton aux appelants qui n'en veulent pas :

```rust,ignore
pub fn surface(contour: &Mesh, et: ElementType, size: Option<f64>) -> Result<Mesh> {
    surface_cancellable(contour, et, size, &NoCancel)
}

pub fn surface_cancellable(
    contour: &Mesh, et: ElementType, size: Option<f64>, cancel: &dyn Cancel,
) -> Result<Mesh> { /* … boucle qui sonde `cancel` … */ }
```

Un programme Rust câble alors ce qu'il veut, sans toucher à Python :

```rust,ignore
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

let stop = Arc::new(AtomicBool::new(false));
// un handler Ctrl+C (crate `ctrlc`), un thread de supervision, un timeout…
let s = stop.clone();
ctrlc::set_handler(move || s.store(true, Ordering::Relaxed)).ok();

let mesh = surface_cancellable(&contour, ElementType::TRI3, Some(0.5), &*stop)?;
// `Deadline::after(Duration::from_secs(10))` marcherait tout aussi bien.
```

**3. Brancher le jeton Python dans la couche FFI** (`src/py`, gatée
`python-api`) — le **seul** endroit où l'interruption Python rencontre le
cœur :

```rust,ignore
struct PySignals<'py>(Python<'py>);

impl Cancel for PySignals<'_> {
    fn check(&self) -> Result<()> {
        self.0.check_signals().map_err(|_| PyrucastError::Interrupted)
    }
}

#[pyfunction]
pub fn surface(py: Python<'_>, contour: PyRef<PyMesh>, /* … */) -> PyResult<PyMesh> {
    let mesh = ops::mesher::surface_cancellable(&contour.inner, et, size, &PySignals(py))?;
    Ok(PyMesh { inner: mesh })
}
```

Le paramètre `py: Python<'_>` est injecté par PyO3 et **n'apparaît pas** dans la
signature Python : `pyrucast.surface(contour, element_type, size=None)` reste
inchangée, mais un `Ctrl+C` l'interrompt désormais.

## Lien avec le parallélisme

Le même `AtomicBool` est le mécanisme naturel pour interrompre un calcul
**parallèle à mémoire partagée** : chaque worker sonde le drapeau, et le thread
principal (seul à détenir le GIL côté Python) le lève quand `check_signals`
détecte le `Ctrl+C`. Poser le trait `Cancel` dès maintenant prépare ce terrain
sans coût supplémentaire.

## À retenir

- L'interruption est **coopérative** : l'opérateur sonde, le frontend décide.
- Le cœur reste **PyO3-free** (`&dyn Cancel`) → utilisable en Rust pur.
- `PyrucastError::Interrupted` → `KeyboardInterrupt` côté Python.
- Sonder une fois par étape grossière ; coût nul pour `NoCancel` (inliné).
