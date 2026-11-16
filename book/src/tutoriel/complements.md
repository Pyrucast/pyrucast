# Compléments

## Éléments structuraux

On choisit le type d'élément par la fonction de modèle appelée : une fonction
`model.<physique>(fes, ...)` par famille d'éléments. La géométrie de la
section (aire `A`, inerties `I`…) se donne comme composantes du champ de
matériau, à côté de `E`.

| pyrucast | primales / duales |
|---|---|
| [`model.truss(fes)`](../mecanique/truss.md) | `u_x, u_y(, u_z)` / `f_x, f_y(, f_z)` |
| [`model.timoshenko(fes)`](../mecanique/timoshenko.md) | en 1-D : `w, theta` / `f_w, m_theta` ; en plan : `u_x, u_y, r_z` / `f_x, f_y, m_z` ; dans l'espace : `u_x, u_y, u_z, r_x, r_y, r_z` / `f_x, f_y, f_z, m_x, m_y, m_z` |

La poutre de Timoshenko demande un espace construit par
`FiniteElementSpace(mesh, interpolation="MODEL_EMBEDDED")` : ses fonctions de
forme dépendent du matériau.

Barre en traction — comparée à la solution analytique
\\( u_x = F \cdot L / (E \cdot A) \\), export du résultat au format VTK :

```python
{{#include ../../../tutoriel/complements.py:barre}}
```

```python
{{#include ../../../tutoriel/complements.py:export}}
```

Les poutres d'[Euler-Bernoulli](../mecanique/bernoulli.md)
(`model.bernoulli`) et les [coques](../mecanique/coques.md) (`model.shell`)
ont leurs pages dans le chapitre de mécanique. Un calcul axisymétrique se pose
sur un `Coords.axisymmetric()`, le plan méridien d'un corps de révolution :
voir [Coordonnées](../coords.md#repère-de-révolution).

Les autres matrices s'assemblent avec [`pyrucast.matrix`](../reference/matrix.md) :
`mass` (matrice de masse cohérente), `lump` (sa version diagonale),
`geometric` (rigidité géométrique, pour le flambage linéarisé) et `tangent`
(matrice tangente cohérente d'une loi non linéaire).

## Mailler en volume

À partir de la page [Maillage](maillage.md), les calculs de ce tutoriel se
font en volume. Les mailleurs qui produisent un volume :

- `pyrucast.mesh.extrude(mesh, direction, n_layers)` — balayage d'un
  maillage selon une direction : `SEG2` → `QUA4` (une surface), `TRI3` →
  `PENTA6`, `QUA4` → `HEX8` ;
- `pyrucast.mesh.revolve(mesh, angle, n_layers, center, axis=None)` — le même
  balayage, mais en rotation autour d'un axe ; un tour complet referme le
  volume engendré ;
- `pyrucast.mesh.sweep_solid(mesh_a, mesh_b, n_layers)` — remplissage entre
  deux maillages de surface `TRI3`/`QUA4` qui se correspondent maille à
  maille (la maille *i* de l'un fait face à la maille *i* de l'autre) ;
- `pyrucast.mesh.triangulate_volume(envelope, size=None)` — remplissage
  `TET4` d'une enveloppe `TRI3` fermée par triangulation de Delaunay 3D ;
  sans `size`, la taille visée est la longueur moyenne des arêtes de
  l'enveloppe.

La page [Maillage](maillage.md) emploie `extrude` et `triangulate_volume`, la
page [Contact](contact.md) `revolve` ; `sweep_solid` n'y sert pas.

## Éléments finis supportés

Catalogue complet : [Éléments finis supportés](../elements/index.md) — 15
types, de `SEG2` à `HEX27`, y compris les versions quadratiques
(`pyrucast.mesh.to_quadratic` passe un maillage linéaire en quadratique).

## Échanges avec les outils extérieurs

pyrucast échange avec trois outils externes :

- **gmsh** — fichier `.msh` (`pyrucast.mesh.read_gmsh`, `read_gmsh_str`) ou
  gmsh ouvert dans le même script Python, dans les deux sens et avec les vues
  (`pyrucast.mesh.from_gmsh`, `pyrucast.export.to_gmsh`) ;
- **MED** (Salome, code_aster) au travers de medcoupling — maillages, groupes,
  champs aux nœuds, aux mailles et aux points de Gauss, séries temporelles
  (`pyrucast.mesh.from_medcoupling`, `pyrucast.export.to_medcoupling`) ;
- **VTK** pour ParaView, dans l'ancien format `.vtk`, en texte ou en
  binaire, et séries temporelles (`pyrucast.export.export_vtk`) — voir la fin
  du script de barre plus haut.

Un calcul se sauvegarde et se relit avec `pyrucast.save` et `pyrucast.load` :
la page [Thermique non linéaire](thermique-non-lineaire.md) écrit un fichier
que relisent les pages
[Dilatation, matériau homogène](mecanique-dilatation.md),
[Dilatation, matériau hétérogène](mecanique-heterogene.md) et
[Plasticité](plasticite.md).

> **Non disponible dans pyrucast.** Pas d'échange Nastran/Abaqus, pas de
> format CSV/Excel dédié pour les listes ou les tables. L'export VTK n'écrit
> que l'ancien format `.vtk` (pas de `.vtu`), et un seul champ par fichier.

## Développer sur pyrucast

pyrucast est un projet Rust ordinaire. Pour ajouter une physique, un élément
fini ou un opérateur, on modifie directement le code Rust puis on relance
`maturin develop --release`. Voir [Développer](../developper.md) — en
particulier [Ajouter une physique](../ajouter-une-physique.md) et
[Ajouter un élément fini](../developper/ajouter-un-element-fini.md).

## Communauté

Le code source est sur [GitHub](https://github.com/Pyrucast/pyrucast), sous
licence MPL-2.0 : on peut y signaler un problème ou y proposer une
modification.
