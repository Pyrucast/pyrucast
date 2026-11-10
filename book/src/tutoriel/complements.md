# Compléments

## Éléments structuraux

pyrucast choisit le type d'élément par l'opérateur de modèle : un opérateur
`model.<physique>(fes, ...)` par famille d'éléments, et des composantes
matériau nommées pour la géométrie de la section.

| pyrucast | primales / duales |
|---|---|
| `model.truss(fes)` | `u_x,u_y(,u_z)` / `f_x,f_y(,f_z)` |
| `model.timoshenko(fes)` | selon la dimension : `w, theta` / `f_w, m_theta` en 1-D, `u_x, u_y, r_z` en plan, six DDL dans l'espace |

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

Pour la matrice de masse cohérente et la rigidité géométrique (flambage
linéarisé), voir [Assemblage par `MatrixKind`](../operateurs/assemblage.md)
— `pyrucast.matrix.mass`/`lump`/`geometric`/`tangent`.

## Mailler en volume

À partir de la page [Maillage](maillage.md), les calculs de ce tutoriel se
font en volume. Les mailleurs de volume de pyrucast :

- `pyrucast.mesh.extrude(mesh, direction, n_couches)` — balayage d'un
  maillage `SEG2`/`TRI3`/`QUA4` selon une direction, dans le même espace de
  coordonnées ;
- `pyrucast.mesh.revolve(mesh, angle, n_couches, centre, axe)` — le même
  balayage, mais en rotation autour d'un axe ; un tour complet referme le
  volume engendré ;
- `pyrucast.mesh.sweep_solid(mesh_a, mesh_b, n_couches)` — balayage entre
  deux profils `TRI3`/`QUA4` non parallèles ;
- `pyrucast.mesh.triangulate_volume(enveloppe, taille)` — remplissage `TET4`
  d'une enveloppe `TRI3` fermée par triangulation de Delaunay 3D.

La page [Maillage](maillage.md) emploie `extrude` et `triangulate_volume`, la
page [Contact](contact.md) `revolve` ; `sweep_solid` n'y sert pas.

## Éléments finis supportés

Catalogue complet : [Éléments finis supportés](../elements/index.md) — 15
types, de `SEG2` à `HEX27`, y compris les versions quadratiques
(`pyrucast.mesh.to_quadratic` passe un maillage linéaire en quadratique).

## Échanges avec les outils extérieurs

pyrucast échange avec trois outils externes :

- **gmsh** — fichier `.msh` (`pyrucast.mesh.read_gmsh`, `read_gmsh_str`) ou
  session vivante, dans les deux sens et avec les vues
  (`pyrucast.mesh.from_gmsh`, `pyrucast.export.to_gmsh`) ;
- **MED** (Salome, code_aster) au travers de medcoupling — maillages, groupes,
  champs aux nœuds, aux mailles et aux points de Gauss, séries temporelles
  (`pyrucast.mesh.from_medcoupling`, `pyrucast.export.to_medcoupling`) ;
- **VTK legacy** pour ParaView, ASCII ou binaire, et séries temporelles
  (`pyrucast.export.export_vtk`) — voir la fin du script de barre plus haut.

Un calcul se sauvegarde et se relit avec `pyrucast.save` et `pyrucast.load` :
la page [Thermique non linéaire](thermique-non-lineaire.md) écrit un fichier
que les pages de mécanique relisent.

> **Non disponible dans pyrucast.** Pas d'échange Nastran/Abaqus, pas de
> format CSV/Excel dédié pour les listes ou les tables. L'export VTK est en
> outre limité au format legacy (pas de `.vtu`) : un champ par fichier.

## Développer sur pyrucast

pyrucast est un projet Rust ordinaire. Pour ajouter une physique, un élément
fini ou un opérateur, on modifie directement le code Rust puis on relance
`maturin develop --release`. Voir [Développer](../developper.md) — en
particulier [Ajouter une physique](../ajouter-une-physique.md) et
[Ajouter un élément fini](../developper/ajouter-un-element-fini.md).

## Communauté

Projet ouvert : voir le fichier `AGENTS.md`/`README` du dépôt pour les
modalités de contribution actuelles.
