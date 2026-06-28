# Objets

Cette partie décrit, un par un, les **objets** du modèle pyrucast : pour
chacun, son **principe** (à quoi il sert, comment il est structuré) et son
**interface** (API Rust et Python). Les objets sont présentés dans leur
**ordre de dépendance** — chacun ne s'appuie que sur les précédents.

```text
Coords ── Node
   │
   ├── Mesh (agrège des SubMesh)
   │      └── FiniteElementSpace (agrège des SubFiniteElementSpace)
   │             ├── ElementField (champ aux points de Gauss)
   │             └── Model (agrège des SubModel : physiques)
   │                    └── Matrix (matrice creuse, sortie d'assemblage)
   └── NodeField (champ aux nœuds)

Evolution (agrège des SubEvolution : valeur tabulée vs variable, interpolée)
```

Deux abstractions transverses traversent toute cette liste et méritent d'être
lues tôt :

- l'[**Agrégat**](aggregate.md) — la grammaire commune (`len`, `[i]`, union
  `|`) de tous les conteneurs « parent / zones » (`Mesh`, `FiniteElementSpace`,
  `Model`, `Matrix`, `NodeField`, `ElementField`, `Evolution`) ;
- le [**Champ**](field.md) (`Field` / `SubField`) — le contrat partagé entre
  `NodeField` et `ElementField` : composantes nommées, `min`/`max`,
  arithmétique scalaire et par composante.

Pour le contexte informatique (store, handles, refcount, motif zone/agrégat),
voir [Aspect informatique](aspect-informatique.md).
