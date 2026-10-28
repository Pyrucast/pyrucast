# Objets

Cette partie décrit, un par un, les **objets** du modèle pyrucast : pour
chacun, son **principe** (à quoi il sert, comment il est structuré) et son
**interface** (API Rust et Python). Les objets sont présentés dans leur
**ordre de dépendance** — chacun ne s'appuie que sur les précédents.

<pre>
<b>Coords</b> ── <b>Node</b>                            <i>le magasin de nœuds, et l'un d'eux</i>
   │
   └── <b>Mesh</b>                               <i>agrège des SubMesh</i>
          ├── <b>NodeField</b>                   <i>champ aux nœuds, sur un support POI1</i>
          └── <b>FiniteElementSpace</b>          <i>agrège des SubFiniteElementSpace</i>
                 ├── <b>ElementField</b>         <i>champ aux points de Gauss</i>
                 └── <b>Model</b>                <i>agrège des SubModel : physiques et contraintes</i>
                        └── <b>Matrix</b>        <i>matrice creuse, sortie d'assemblage</i>

<b>Evolution</b>                                 <i>agrège des SubEvolution : valeur tabulée selon une variable</i>
</pre>

Une distinction structure toute cette liste et se lit dans l'arborescence :
`Coords` est le **magasin** (`src/coords.rs`), `Node`, `Cell` et `Element`
sont des **atomes** insécables (`src/atoms/`), et tout le reste — les sept agrégats et
leurs vues `Sub*` — sont des **conteneurs** divisibles (`src/containers/`),
les seuls qui puissent être le sujet d'un opérateur. Voir
[Conventions](conventions.md).

Deux abstractions transverses traversent la liste et méritent d'être
lues tôt :

- l'[**Agrégat**](aggregate.md) — la grammaire commune (`len`, `[i]`, union
  `|`) de tous les conteneurs « parent / zones » (`Mesh`, `FiniteElementSpace`,
  `Model`, `Matrix`, `NodeField`, `ElementField`, `Evolution`) ;
- le [**Champ**](field.md) (`Field` / `SubField`) — le contrat partagé entre
  `NodeField` et `ElementField` : composantes nommées, `min`/`max`,
  arithmétique scalaire et par composante.

Pour le contexte informatique (durée de vie des objets, motif zone/agrégat),
voir [Aspect informatique](aspect-informatique.md) ; pour ce qu'il y a dessous
côté Rust (`Handle<T>`, compteurs de références), [Modèle
mémoire](memory-model.md). Pour les écrire dans un
fichier et les relire tels qu'ils étaient, [Sauvegarde et
relecture](sauvegarde.md).
