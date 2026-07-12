# Détails des physiques

Chaque **physique** est une variante de [`SubModel`](model.md) : elle déclare
ses variables (primales / duales), son matériau, et sait assembler sa rigidité
`K` (et, le cas échéant, intégrer son comportement `COMP`). Le `Model`
orchestre, mais ne porte aucune logique physique — voir
[Modèle physique](model.md) pour la mécanique générique (DOFs, assemblage,
chargements, solveur).

Cette partie décrit, physique par physique : les **équations résolues**, la
**mise en donnée** (exemple Rust testé via `{{#include}}`), et un **exemple
Python**.

- [Conduction thermique](thermique.md) — `-∇·(k∇T) = 0`, l'exemple canonique,
  et la [convection de surface](thermique.md#convection-de-surface-robin--film)
  (Robin / film, `q·n = h(T − T_ext)`).
- [Mécanique](mecanique.md) — barre, élasticité linéaire, poutres et
  portiques :
  - [Barre / treillis](mecanique/truss.md)
  - [Élasticité linéaire](mecanique/elasticite.md)
  - [Plasticité parfaite (von Mises)](mecanique/plasticite.md)
  - [Endommagement de Mazars](mecanique/mazars.md)
  - [Poutre de Timoshenko](mecanique/timoshenko.md)
  - [Portique 2D](mecanique/portique.md)
  - [Cadre 3D](mecanique/cadre3d.md)
- [Contraintes](contraintes.md) — conditions limites imposées par
  multiplicateurs de Lagrange :
  - [Dirichlet](contraintes/dirichlet.md)
  - [Multi-points (MPC)](contraintes/mpc.md)
  - [Baignage (embedded)](contraintes/embedded.md)
  - [Contact (nœud-surface)](contraintes/contact.md)

Ce regroupement est la **nature physique** (`Physics`) que chaque variante
déclare : `Thermal` (conduction), `Mechanical` (barre → cadre 3D) et
`Constraint` (les contraintes de Lagrange). On sélectionne les sous-modèles
d'une nature avec `model.filter(Physics::Mechanical)` (et les blocs d'une
matrice avec `k.filter(...)`) — voir
[Nature physique et filtrage](model.md#nature-physique-et-filtrage).

Pour **ajouter** une physique, voir [Ajouter une physique](ajouter-une-physique.md).
