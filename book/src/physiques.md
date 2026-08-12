# Détails des physiques

Chaque **physique** est une variante de [`SubModel`](model.md) : elle déclare
ses variables (primales / duales), son matériau, et sait assembler sa rigidité
`K` (et, le cas échéant, intégrer son comportement `COMP`). Le `Model`
orchestre, mais ne porte aucune logique physique — voir
[Modèle physique](model.md) pour la mécanique générique (DOFs, assemblage,
chargements, solveur).

Chaque page suit le **même plan standard**, dans cet ordre :

1. **Introduction** — nature du modèle, éléments et degrés de liberté.
2. **Équations continues résolues** — forme forte (et faible) du problème.
3. **Forme discrétisée** — opérateurs discrets (`B`, `N`) et expressions des
   matrices élémentaires (`K`, masse/capacité, tangente…).
4. **Variables et matériau** — noms primal/dual, composantes matériau,
   comportement (`COMP`) et, le cas échéant, état interne.
5. **Mise en donnée (Rust, testé)** — exemple Rust exécuté via `{{#include}}`.
6. **Exemple Python** — l'équivalent haut niveau.
7. **Compléments** (facultatif) — extensions propres au modèle (masse &
   rigidité géométrique, thermomécanique, convection, régime transitoire…).

Certaines physiques omettent une rubrique sans objet (p. ex. pas d'exemple Rust
dédié pour un modèle piloté par le comportement) ; l'ordre reste identique.

- [Conduction thermique](thermique.md) — `-∇·(k∇T) = 0`, l'exemple canonique,
  la [conduction orientée](thermique.md#conduction-orthotrope-et-anisotrope)
  (tenseur `K`) et la
  [convection de surface](thermique.md#convection-de-surface-robin--film)
  (Robin / film, `q·n = h(T − T_ext)`), plus le
  [rayonnement à l'infini](thermique.md#rayonnement-à-linfini-stefan-boltzmann)
  (`q·n = σε(T⁴ − T_∞⁴)`, non linéaire, à tangente cohérente).
- [Diffusion (loi de Fick)](diffusion.md) — `∇·(D∇c) = 0`, concentration `c` et
  flux de matière `j` ; même opérateur que la conduction, nature distincte. Avec
  le [transfert d'interface](diffusion.md#transfert-à-travers-une-interface)
  `j·n = h(c₁ − c₂)`, qui laisse le champ sauter entre deux corps.
- [Mécanique](mecanique.md) — barre, élasticité linéaire, poutres et
  portiques :
  - [Barre / treillis](mecanique/truss.md)
  - [Élasticité linéaire](mecanique/elasticite.md)
  - [Élasticité orthotrope et anisotrope](mecanique/orthotropie.md)
  - [Pression suiveuse](mecanique/pression-suiveuse.md)
  - [Plasticité parfaite (von Mises)](mecanique/plasticite.md)
  - [Lois d'écoulement plastique](mecanique/lois-plastiques.md) — écrouissage
    isotrope, Drucker-Prager, Ottosen
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
déclare : `Thermal` (conduction), `Radiation` (rayonnement, en plus de
`Thermal`), `Diffusion` (Fick), `Mechanical` (barre →
cadre 3D) et `Constraint` (les contraintes de Lagrange). On sélectionne les sous-modèles
d'une nature avec `model.filter(Physics::Mechanical)` (et les blocs d'une
matrice avec `k.filter(...)`) — voir
[Nature physique et filtrage](model.md#nature-physique-et-filtrage).

Pour **ajouter** une physique, voir [Ajouter une physique](ajouter-une-physique.md).
