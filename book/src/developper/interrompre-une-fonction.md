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

## Le principe : un jeton, choisi par l'appelant

L'écueil serait de faire appeler `Python::check_signals` directement par
l'opérateur — cela couplerait le **cœur de calcul** à PyO3 et casserait
l'usage en Rust pur (cf. [Compilation et tests](../compilation.md), build sans
`python-api`).

À la place, l'opérateur reçoit un **jeton d'interruption** abstrait — le trait
`Cancel` du module `interrupt` — qu'il interroge périodiquement. C'est
l'**appelant** qui décide *comment* l'interruption est signalée :

| Appelant | Jeton | Déclencheur |
|---|---|---|
| Rust pur | `NoCancel` | jamais |
| Rust pur | `()` | jamais |
| Rust pur | `AtomicBool` | un autre thread, ou un gestionnaire de signal, lève le drapeau |
| Rust pur | `Deadline` | dépassement d'un délai |
| Python | `PySignals` (dans `src/py`, feature `python-api`) | `Ctrl+C` via `Python::check_signals` |

Le trait est du **Rust pur, sans PyO3** :

```rust,ignore
{{#include ../../../src/interrupt.rs:trait}}
```

L'erreur renvoyée est `PyrucastError::Interrupted`. Sa conversion vers Python
(feature `python-api`) produit une vraie **`KeyboardInterrupt`**, pas un
`RuntimeError` générique. Le cœur, lui, ne voit que `&dyn Cancel` : il reste
PyO3-free et utilisable depuis un programme Rust.

## Implémenter l'interruption dans un opérateur

Trois gestes.

**1. Faire passer le jeton et le sonder.** Le cœur de calcul prend un
`&dyn Cancel` et l'interroge à chaque tour de boucle (un tour = un événement
*grossier* — un élément, une couche, une itération de solveur — pour que le
coût d'un `check` soit négligeable) :

```text
pub fn mon_operateur(/* arguments */, cancel: &dyn Cancel) -> Result<Mesh> {
    loop {
        cancel.check()?;          // ← point d'interruption
        // une étape de travail
    }
}
```

> Sonder une fois par étape grossière suffit ; inutile d'espacer les sondes par
> un compteur si chaque tour fait déjà un travail substantiel.

**2. Exposer deux formes côté Rust** — une simple et une interruptible — pour
ne pas imposer un jeton aux appelants qui n'en veulent pas :

```text
pub fn triangulate_surface(
    contour: &Mesh, element_type: ElementType, target_size: Option<f64>,
) -> Result<Mesh> {
    triangulate_surface_cancellable(contour, element_type, target_size, &NoCancel)
}

pub fn triangulate_surface_cancellable(
    contour: &Mesh, element_type: ElementType, target_size: Option<f64>,
    cancel: &dyn Cancel,
) -> Result<Mesh> { /* la boucle qui sonde `cancel` */ }
```

Un programme Rust câble alors ce qu'il veut, sans toucher à Python :

```rust,ignore
{{#include ../../../tests/doc_memoire.rs:annulation}}
```

**3. Brancher le jeton Python dans le binding** (`src/py`, feature
`python-api`) — le **seul** endroit où l'interruption Python rencontre le
cœur :

```rust,ignore
{{#include ../../../src/py/signals.rs:pysignals}}
```

Le paramètre `py: Python<'_>` est injecté par PyO3 et **n'apparaît pas** dans la
signature Python : `pyrucast.mesh.triangulate_surface(contour, element_type, size=None)`
ne le montre pas, et un `Ctrl+C` l'interrompt.

## Interruption et parallélisme

Le jeton est sondé par le thread appelant, entre deux étapes. Une région
parallèle (rayon) ne le sonde pas : `PySignals` tient le `Python<'py>` du
thread qui détient le GIL, et ne passe pas aux autres threads. Un `Ctrl+C`
pendant une région parallèle prend donc effet au point de sonde suivant, une
fois la région terminée.

## À retenir

- L'interruption est **coopérative** : l'opérateur sonde, l'appelant décide.
- Le cœur reste **sans PyO3** (`&dyn Cancel`) → utilisable en Rust pur.
- `PyrucastError::Interrupted` → `KeyboardInterrupt` côté Python.
- Sonder une fois par étape grossière : un appel indirect par étape, coût
  négligeable devant le travail de l'étape.
