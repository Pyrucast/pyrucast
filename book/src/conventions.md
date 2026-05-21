# Conventions & philosophie

## Erreurs

Toute l'API publique renvoie `pyrucast::Result<T>`, alias de `Result<T, PyrucastError>`.

`PyrucastError` est l'unique type d'erreur de la librairie. Côté Python, il est converti automatiquement en `RuntimeError`.

## Affichage : `Debug` vs `Display`

Chaque objet du modèle implémente deux traits :

- `Debug` — vue **structurelle** : utile pour le développement, exposée en Python via `__repr__`.
- `Display` — vue **résumée** orientée utilisateur EF, façon listing cast3m, exposée en Python via `__str__`.

Le binding PyO3 branche ces deux vues sur les dunder methods Python correspondantes.

## Sérialisation : un seul mécanisme

Le trait `Persist` (implémenté automatiquement pour tout type `serde::Serialize + DeserializeOwned`) produit un format binaire **portable Linux ↔ Windows**. Ce socle unique sert à la fois :

- au **swap disque** (slot par slot, géré par le Store) ;
- à la **sauvegarde fichier** (graphe d'objets d'une `Session`, dans un conteneur versionné).

## Definition of Done par objet

Un objet n'est considéré comme terminé que lorsque les six points suivants sont verts :

1. Struct Rust vivant dans le Store (`Handle<T>` typé).
2. `Debug` (structure) + `Display` (résumé).
3. Tests unitaires Rust + doctests sur tout l'API public.
4. Binding PyO3 (`__repr__` / `__str__`).
5. Tests Python (pytest).
6. Chapitre de cette documentation.

## Dépendances approuvées

Le socle figé est : `pyo3`, `maturin`, `mdbook`, `serde`, `bincode`. Toute autre dépendance, Rust ou Python, requiert un accord explicite.
