# Sommaire des opérateurs

Les **opérateurs** sont les fonctions libres de pyrucast : elles **croisent
des conteneurs** (maillage + champ, espace EF + champ…) ou appartiennent à une
famille d'opérateurs, par opposition aux **méthodes** qui restent sur un seul
conteneur (cf. [Conventions](conventions.md)). Chacune vit dans le module qui
porte le nom du **conteneur qu'elle rend** : `pyrucast.matrix.stiffness` rend
une matrice, `pyrucast.node_field.positions` un champ aux nœuds.

Cette page les range **par thème**, ce qui ne recoupe pas toujours le module.
Chaque ligne donne le prototype abrégé — les arguments obligatoires, `…` s'il
en existe de facultatifs — et la première phrase de la documentation de la
fonction. Le prototype mène à la **référence**, où se trouvent le prototype
complet, la description de chaque argument et un exemple.

La référence est écrite en anglais : elle est **fabriquée à partir de la
documentation embarquée dans pyrucast**, celle que montrent `help()` et
l'éditeur. Elle ne peut donc pas s'en écarter.

La plupart de ces opérations sont **aussi** des méthodes de leur sujet, pour
permettre le chaînage — `maillage.border().consolidate()` plutôt que
`mesh.consolidate(mesh.border(maillage))`. La référence le signale sous chaque
fonction ; la règle qui décide lesquelles tient dans les trois conditions de
[Conventions](conventions.md#le-verbe-exposé-aussi-en-méthode).

## Maillage

### Lignes et courbes

{{#pysummary mesh.line mesh.arc mesh.circle}}

### Surfaces et volumes réglés

{{#pysummary
mesh.transfinite mesh.sweep mesh.extrude mesh.revolve}}

### Mailleurs automatiques

{{#pysummary
mesh.triangulate_surface mesh.pave_surface mesh.grid_surface mesh.grid_surface2
mesh.triangulate_volume mesh.pave_volume}}

Voir aussi [Mailler une géométrie](mailler.md).

### Améliorer un maillage

{{#pysummary mesh.regularize mesh.cleanup mesh.merge_triangles mesh.merge_nodes}}

### Déplacer et copier

{{#pysummary
mesh.copy mesh.translate mesh.rotate
mesh.symmetry_point mesh.symmetry_line mesh.symmetry_plane
mesh.project_on_sphere}}

### Changer d'élément

{{#pysummary mesh.convert mesh.to_quadratic mesh.to_poi1}}

### Extraire une partie

{{#pysummary
mesh.border mesh.skin mesh.elements_on mesh.intersection mesh.difference
mesh.barycenter
mesh.poi1_from_nodes mesh.from_live_nodes mesh.select}}

### Ordonner et regrouper

{{#pysummary mesh.orient mesh.invert mesh.chain mesh.consolidate}}

### Choisir des nœuds par région

{{#pysummary
mesh.points_on_line
mesh.points_on_plane mesh.points_below_plane
mesh.points_on_sphere mesh.points_in_sphere
mesh.points_on_cylinder mesh.points_in_cylinder
mesh.points_on_cone mesh.points_in_cone
mesh.points_on_torus mesh.points_in_torus}}

## Physique

Les pages [Détails des physiques](physiques.md) donnent les équations de
chaque modèle.

### Thermique et diffusion

{{#pysummary model.heat_conduction model.fick}}

### Échanges et chargements

{{#pysummary
model.flux model.boundary_transfer model.interface_transfer model.radiation}}

### Milieux continus

{{#pysummary
model.elasticity
model.plasticity_perfect model.plasticity_isotropic
model.drucker_prager model.ottosen model.gurson
model.creep_norton model.creep_lemaitre model.creep_blackburn
model.viscoplasticity_chaboche model.viscoplasticity_lemaitre_chaboche
model.mazars model.damage_tc model.damage_sic_sic}}

### Barres, poutres et coques

{{#pysummary model.truss model.bernoulli model.timoshenko model.shell}}

### Contraintes

{{#pysummary
model.dirichlet model.mpc model.embedded model.contact model.update_contact}}

## Champs

### Matériau

{{#pysummary
element_field.material_field element_field.material_field_per_sub_model
element_field.sub_material_field}}

### Géométrie et configurations

{{#pysummary
node_field.positions coords.set coords.displace coords.displace_from}}

### Dériver et interpoler

{{#pysummary
element_field.gradient node_field.divergence element_field.interp_to_gauss}}

### Déformations

{{#pysummary
element_field.deformation element_field.strain element_field.thermal_strain
element_field.beam_deformation element_field.shell_deformation}}

### Contraintes et comportement

{{#pysummary
element_field.initial_state element_field.integrate_behavior
element_field.elastic_stress element_field.elastic_strain
element_field.von_mises element_field.stress_measure
element_field.rotate_frame}}

### Restreindre, projeter, réunir

{{#pysummary
node_field.restrict node_field.restrict_like finite_element_space.restrict
element_field.restrict element_field.restrict_like node_field.project
node_field.merge node_field.consolidate element_field.consolidate
node_field.mask element_field.mask
node_field.to_nodes}}

### Multiplier des tenseurs

{{#pysummary
element_field.tensor_dot element_field.tensor_ddot}}

### Mesurer

{{#pysummary
measure.integral element_field.cell_integral
measure.xtx measure.xty field.psca
evolution.along}}

### Fonctions mathématiques

{{#pysummary
field.abs field.sqrt field.exp field.log field.log10
field.cos field.sin field.tan field.cosh field.sinh field.tanh}}

## Assemblage et résolution

### Matrices

{{#pysummary
matrix.stiffness matrix.mass matrix.geometric matrix.tangent matrix.lump}}

### Seconds membres

{{#pysummary node_field.internal_forces node_field.external_forces}}

### Résolution

{{#pysummary solver.solve}}

## Échanges avec l'extérieur

### Lire un maillage

{{#pysummary
mesh.read_gmsh mesh.read_gmsh_str mesh.from_gmsh mesh.from_medcoupling
mesh.from_arrays}}

### Écrire un maillage et ses champs

{{#pysummary
export.export_vtk export.to_gmsh export.to_medcoupling export.to_arrays}}

Voir aussi [Visualisation](visualization.md).

### Correspondances avec un autre code

{{#pysummary
mesh.element_type_from_gmsh mesh.gauss_to_external mesh.match_gauss}}

### Sauvegarder et relire

{{#pysummary save load spill_stats}}

Voir aussi [Sauvegarde et relecture](sauvegarde.md).

## La chaîne typique d'un calcul

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
   element_field ── strain ──► element_field.behavior ── integrate ──► (efforts)
                                     │
                                     ▼
                                   viz ── plot
```
