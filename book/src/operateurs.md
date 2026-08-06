# Détail des opérateurs

Les **opérateurs** sont les fonctions libres de pyrucast : elles **croisent
des conteneurs** (maillage + champ, espace EF + champ…) ou appartiennent à une
famille d'opérateurs, par opposition aux **méthodes** qui restent sur un seul
conteneur (cf. [Conventions](conventions.md)). Côté Rust elles vivent sous
`src/ops/<module>`, où le module porte le nom du **conteneur produit** ; côté
Python elles sont exposées dans le sous-module de même nom
(`pyrucast.node_field.positions`, `pyrucast.matrix.stiffness`, …).

Les chapitres qui suivent sont organisés **par sujet**, ce qui ne recoupe pas
toujours le module d'implémentation — la colonne de gauche donne la
correspondance.

| Module Rust | Chapitre | Contenu |
|---|---|---|
| `ops::mesh` | [Maillage](operateurs/maillage.md) | `line`, `circle`, `arc`, `extrude`, `revolve`, `sweep`, `transfinite` (DALL), `sweep_solid`, `translate`, `rotate`, `symmetry_point`, `symmetry_line`, `symmetry_plane`, `triangulate_surface`, `pave_surface`, `grid_surface`, `triangulate_volume`, `pave_volume`, `border`, `skin`, `orient`, `invert`, `chain`, `elements_on`, sélection de nœuds par région (`points_in_sphere`, `points_on_plane`, `points_in_cylinder`, `points_on_cone`, `points_on_torus`…), `merge_nodes`, `read_gmsh`, `to_poi1`, `to_quadratic`, `convert`, `barycenter`, `mesh.consolidate`… |
| `ops::element_field` | [Construction](operateurs/construction.md) | champs matériau (`material_field`…) |
| `ops::coords` | [Champs](operateurs/champs.md) | `set`, `displace` — les deux seuls opérateurs qui écrivent la géométrie |
| `ops::measure` | [Champs](operateurs/champs.md) | `integral` / `integral_element` (`∫ f dΩ`), `xtx` / `xty` (produits scalaires globaux) |
| `ops::geom` | [Géométrie](operateurs/geometrie.md) | `locate_points` (mapping inverse, baignage), `project_points` (projection sur surface, contact) — internes, pas exposées à Python |
| `ops::node_field`, `ops::element_field`, `ops::field` | [Champs](operateurs/champs.md) | `positions`, `gradient`, `divergence`, `deformation`, `beam_deformation`, `frame_deformation`, `interp_to_gauss` (nœuds → Gauss), `thermal_strain` (déformation thermique `EPTH`), `restrict`, `restrict_like` (reprojection sur le support d'un champ cible), `select`, `mask`, `filter_components` / `rename_component` (extraction et renommage de composantes, `EXCO`), `merge`, `node_field.consolidate` / `element_field.consolidate`, `integral` / `integral_element` (intégrale `∫ f dΩ`), `xty` / `xtx` (produits scalaires globaux) / `psca` (produit scalaire nœud par nœud), maths élément par élément (`abs`, `sqrt`, `exp`, `cos`…)… |
| `ops::matrix` | [Assemblage](operateurs/assemblage.md) | `stiffness`, `mass`, rigidité géométrique `geometric`, tangente cohérente `tangent`, concentration `lump`, composition `assemble` (réassemble depuis les blocs seuls, sans `Model`), chargement réparti `flux`, forces internes `internal_forces` / `internal_forces_continuum` (le `BSIG`, `∫ Bᵀ σ`) |
| `ops::element_field::behavior` | [Comportement](operateurs/comportement.md) | `integrate_behavior` (le `COMP`) |
| `ops::solver` | [Solveur](operateurs/solveur.md) | `solve` (LU creux, Lagrange), `solve_eliminate` (condensation MPC), `solve_unilateral` (actif/inactif, relations unilatérales) |
| `ops::export` | [Visualisation](visualization.md) | `export_vtk` (maillage / champ → VTK pour ParaView) |
| `src/viz` | [Visualisation](visualization.md) | tracé des maillages, coloration par champ |

Le découpage est **par conteneur produit** : `gradient(field, fespace)` rend un
`ElementField`, il vit donc dans `ops::element_field` à côté de `deformation`,
et non avec `divergence` (qui rend un champ nodal). Une opération se range par
sa sortie, jamais par son entrée. Le binding Python reste un miroir 1:1 — voir
[Correspondance Rust ↔ Python](correspondance-rust-python.md).

La **fonction libre est la forme canonique** : c'est elle qui est documentée
dans les chapitres qui suivent. La plupart de ces opérations sont **aussi**
des méthodes de leur sujet, pour permettre le chaînage —
`maillage.border().consolidate()` plutôt que
`mesh.consolidate(mesh.border(maillage))`. La règle qui décide lesquelles, et
ses exclusions, tient dans les trois conditions de
[Conventions](conventions.md#le-verbe-exposé-aussi-en-méthode).

La **chaîne typique** d'un calcul enchaîne ces opérateurs :

```text
mesh ──► (Mesh) ──► FiniteElementSpace
                                 │
element_field ── material_field ─┤
                                 ▼
matrix ── stiffness ──────────► (Matrix) ──┐
node_field ── flux/positions ──► (RHS) ──┤
                                           ▼
                          solver ── solve ──► (NodeField solution)
                                           │
   element_field ── deformation ──► element_field.behavior ── integrate ──► (efforts)
                                     │
                                     ▼
                                   viz ── plot
```
