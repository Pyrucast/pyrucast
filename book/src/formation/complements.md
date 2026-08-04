# Compléments

## Éléments structuraux

Cast3M sélectionne le type d'élément dans `MODE` (`'POUT'`, `'TIMO'`,
`'DKT'`, `'COQ4'`…) et ses caractéristiques géométriques dans `MATE`
(`'SECT'`, `'INRY'`, `'EPAI'`…). pyrucast suit le même principe : un
constructeur `Model.<physique>(fes, ...)` par famille d'éléments, des
composantes matériau nommées portant la géométrie de la section.

| pyrucast | Cast3M | primales / duales |
|---|---|---|
| `Model.truss(fes)` | `MODE ... 'BARR'` | `u_x,u_y(,u_z)` / `f_x,f_y(,f_z)` |
| `Model.timoshenko(fes)` | `MODE ... 'TIMO'` | `w, theta` / `f_w, m_theta` |
| `Model.frame(fes)` | `MODE ... 'POUT'` (2D, portique) | `u_x, u_y, rz` / `f_x, f_y, m_z` |
| `Model.frame3d(fes)` | `MODE ... 'POUT'` (3D, cadre) | 6 primales/duales (translations + rotations) |

Barre en traction — comparée à la solution analytique
\\( u_x = F \cdot L / (E \cdot A) \\), export du résultat au format VTK :

```python
{{#include ../../../formation/complements.py:barre}}
```

```python
{{#include ../../../formation/complements.py:export}}
```

> **Non disponible dans pyrucast.** Pas d'élément coque (Cast3M `DKT`,
> `COQ4`, `COQ2`) — la mécanique 2D/3D de pyrucast reste un continuum
> (contraintes planes, déformations planes, 3D massif) ou des éléments
> structuraux filaires (barre, poutre). Pas de mode axisymétrique
> (Cast3M `OPTI 'MODE' 'AXIS'`) ni de configuration purement 1D
> (`OPTI 'DIME' 1`) — `Coords` est toujours 2D ou 3D cartésien.

Pour la matrice de masse cohérente et la rigidité géométrique (flambage
linéarisé), voir [Assemblage par `MatrixKind`](../operateurs/assemblage.md)
— `pyrucast.matrix.mass`/`lump`/`geometric`/`tangent`, l'équivalent
Cast3M `MASS`/`LUMP`/`KSIG`/`KTAN`.

## Aller plus loin en 3D

Cette formation reste en 2D, mais rien n'empêche de reprendre la même
plaque trouée en volume :

- `pyrucast.mesh.extrude(mesh, direction, n_couches)` — balayage d'un
  maillage `SEG2`/`TRI3`/`QUA4` selon une direction, dans le même espace de
  coordonnées (Cast3M `TRAN`/`VOLU 'TRAN'`) ;
- `pyrucast.mesh.revolve(mesh, angle, n_couches, centre, axe)` — le même
  balayage, mais en rotation autour d'un axe ; un tour complet referme le
  volume engendré (Cast3M `ROTA`/`VOLU 'ROTA'`) ;
- `pyrucast.mesh.sweep_solid(mesh_a, mesh_b, n_couches)` — balayage entre
  deux profils `TRI3`/`QUA4` non parallèles (Cast3M `REGL` + `VOLU`) ;
- `pyrucast.mesh.triangulate_volume(enveloppe, taille)` — remplissage `TET4`
  d'une enveloppe `TRI3` fermée par triangulation de Delaunay 3D (Cast3M
  `VOLU` par remplissage).

Aucun de ces quatre n'est mis en œuvre dans les scripts de cette formation.

## Éléments finis supportés

Catalogue complet : [Éléments finis supportés](../elements/index.md) — 14
types, de `POI1` à `HEX27`, y compris les versions quadratiques
(`pyrucast.mesh.to_quadratic`, l'équivalent Cast3M `CHAN 'TRI6' ...`).

## Échanges avec les outils extérieurs

pyrucast ne parle, à ce jour, que deux formats externes :

- **import** de maillage **Gmsh** (`pyrucast.mesh.read_gmsh`,
  `read_gmsh_str`) — un dictionnaire `nom de région → Mesh` ;
- **export VTK legacy** (`pyrucast.export.export_vtk`), lisible par
  ParaView — voir la fin du script de barre plus haut.

> **Non disponible dans pyrucast.** Pas d'échange Nastran/Abaqus/MED/Salomé,
> pas de format CSV/Excel dédié pour les listes ou les tables (Cast3M
> `SORT 'EXCE'`/`LIRE 'CSV'`), pas de format XDR de sauvegarde/restitution
> (Cast3M `OPTI 'SAUV'`/`OPTI 'REST'`) — un script pyrucast reconstruit
> toujours son état depuis son code, il ne le sérialise pas sur disque.
> L'export VTK est en outre limité au format legacy ASCII : un maillage et
> un champ par fichier, pas de série temporelle (pas d'équivalent PVD).

## Développer sur pyrucast

pyrucast est un projet Rust ordinaire : pas de procédures Gibiane à écrire
dans un dossier `procedur/`, pas de compilation de sources Esope. Pour
ajouter une physique, un élément fini ou un opérateur, on modifie
directement le code Rust puis on relance `maturin develop --release`. Voir
[Développer](../developper.md) — en particulier
[Ajouter une physique](../ajouter-une-physique.md) et
[Ajouter un élément fini](../developper/ajouter-un-element-fini.md), qui
jouent le rôle des chapitres Cast3M sur les sources Esope.

## Communauté

Projet ouvert : voir le fichier `AGENTS.md`/`README` du dépôt pour les
modalités de contribution actuelles.
