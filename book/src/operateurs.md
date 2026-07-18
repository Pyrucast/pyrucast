# Détail des opérateurs

Les **opérateurs** sont les fonctions libres de pyrucast : elles **croisent
des conteneurs** (maillage + champ, espace EF + champ…) ou appartiennent à une
famille d'opérateurs, par opposition aux **méthodes** qui restent sur un seul
conteneur (cf. [Conventions](conventions.md)). Côté Rust elles vivent sous
`src/ops/<thème>` ; côté Python elles sont exposées **à plat** au top-level
(`pyrucast.field.coordinates`, `pyrucast.assemble.stiffness`, …).

Cette partie suit **exactement les thèmes des modules `ops/`** : un chapitre
par module, plus la visualisation (qui vit sous `src/viz/`).

| Module Rust | Chapitre | Contenu |
|---|---|---|
| `ops::mesher` | [Maillage](operateurs/maillage.md) | `line_seg2`, `circle_seg2`, `extrude`, `sweep_qua4`, `sweep_solid`, `translate`, `rotate`, `fill_surface`, `surface`, `volume`, `contour`, `elements_on`, `merge_nodes`, `read_gmsh`, `to_poi1`, `to_quadratic`, `barycenter`, `consolidate`… |
| `ops::build` | [Construction](operateurs/construction.md) | champs matériau (`material_field`…) |
| `ops::geom` | [Géométrie](operateurs/geometrie.md) | `locate_points` (mapping inverse, baignage), `project_points` (projection sur surface, contact), `nearest_node` (nœud le plus proche d'un point) |
| `ops::field` | [Champs](operateurs/champs.md) | `coordinates`, `gradient`, `divergence`, `deformation`, `beam_deformation`, `interp_to_gauss` (nœuds → Gauss), `thermal_strain` (déformation thermique `EPTH`), `restrict`, `restrict_like` (reprojection sur le support d'un champ cible), `select`, `mask`, `filter_components` / `rename_component` (extraction et renommage de composantes, `EXCO`), `merge`, `consolidate`, `integral` / `integral_element` (intégrale `∫ f dΩ`), `xty` / `xtx` (produits scalaires globaux) / `psca` (produit scalaire nœud par nœud), maths élément par élément (`abs`, `sqrt`, `exp`, `cos`…)… |
| `ops::assemble` | [Assemblage](operateurs/assemblage.md) | `stiffness`, `mass`, rigidité géométrique `geometric`, tangente cohérente `tangent`, concentration `lump`, composition `assemble` (réassemble depuis les blocs seuls, sans `Model`), chargement réparti `flux`, forces internes `internal_forces` / `internal_forces_continuum` (le `BSIG`, `∫ Bᵀ σ`) |
| `ops::behavior` | [Comportement](operateurs/comportement.md) | `integrate_behavior` (le `COMP`) |
| `ops::solver` | [Solveur](operateurs/solveur.md) | `solve` (LU creux, Lagrange), `solve_eliminate` (condensation MPC), `solve_unilateral` (actif/inactif, relations unilatérales) |
| `ops::export` | [Visualisation](visualization.md) | `export_vtk` (maillage / champ → VTK pour ParaView) |
| `src/viz` | [Visualisation](visualization.md) | tracé des maillages, coloration par champ |

Le découpage est **par thème, pas par conteneur** : `gradient(field, fespace)`
vit à côté de `divergence`, pas dans `mesh.rs` *ou* `node_field.rs`. Le binding
Python reste un miroir 1:1 — voir
[Correspondance Rust ↔ Python](correspondance-rust-python.md).

La **chaîne typique** d'un calcul enchaîne ces opérateurs :

```text
mesher ──► (Mesh) ──► FiniteElementSpace
                          │
build ── material_field ──┤
                          ▼
assemble ── stiffness ──► (Matrix) ──┐
field ── flux/coordinates ──► (RHS) ─┤
                                     ▼
                          solver ── solve ──► (NodeField solution)
                                     │
                          field ── deformation ──► behavior ── integrate ──► (efforts)
                                     │
                                     ▼
                                   viz ── plot
```
