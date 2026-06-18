# Dirichlet

La condition de **Dirichlet** impose la valeur d'une variable primale,
`u(n) = u_d`, sur un ensemble de nœuds. C'est une [contrainte](../contraintes.md)
imposée par **multiplicateurs de Lagrange** : aucun matériau, aucune loi de
comportement. Elle ne crée **aucun nœud** et ne mute jamais le `Coords`.

Implémentation : `src/models/dirichlet.rs` ; constructeur `Model::dirichlet(…)`.

## Deux maillages fournis par l'utilisateur

L'utilisateur fournit **deux maillages POI1** :

- **`imposed_mesh`** — les nœuds contraints (partagés avec la physique cible) ;
- **`multiplier_mesh`** — le support des **multiplicateurs**, apparié
  élément-par-élément avec `imposed_mesh` (même structure de sous-maillage,
  même nombre de cellules par paire).

On fabrique typiquement `multiplier_mesh` depuis `imposed_mesh` avec le mesher
générique [`barycenter`](../operateurs/maillage.md) (des nœuds neufs colocalisés
au centre de gravité de chaque cellule). Mais l'utilisateur reste libre :
nœuds colocalisés, décalés, ou même réutiliser les nœuds contraints eux-mêmes.

## Quatre noms de variables

Deux sont requis, deux sont déduits et **surchargeables** :

| rôle | nom | fourniture |
|---|---|---|
| variable imposée (primale de la **cible**) | `imposed_variable` (ex `"T"`) | requis |
| duale de la **cible** (ligne où atterrit la réaction `Cᵀ`) | `target_dual` (ex `"q"`) | requis |
| primale propre = multiplicateur (inconnue du système) | `multiplier`, défaut `lambda_<imposed_variable>` | déduit |
| duale propre = ligne de contrainte + **slot** où l'utilisateur écrit `u_d` | `imposed_value`, défaut `imposed_<imposed_variable>` | déduit |

Signature complète :

```text
Model.dirichlet(imposed_variable, target_dual, imposed_mesh, multiplier_mesh,
                multiplier=None, imposed_value=None)
```

## Les deux blocs unité

À l'assemblage, Dirichlet contribue **une paire de blocs unité par
sous-maillage**, chacun marqué **non-symétrique** (seule l'union `C ∪ Cᵀ` l'est
— propriété **globale** du système point-selle ; cf. le drapeau `symmetric` de
la [Matrice](../matrix.md)) :

- **bloc C** : `(multiplier_node, imposed_value) × (imposed_node, imposed_variable) = 1`
- **bloc Cᵀ** : `(imposed_node, target_dual) × (multiplier_node, multiplier) = 1`

Le bloc `C` exprime la relation `u(n) = u_d` ; le bloc `Cᵀ` réinjecte la
réaction dans l'équation de la physique cible (ligne `target_dual`).

## Valeur imposée et réaction

- La **valeur imposée** `u_d` n'est **pas** stockée dans le SubModel :
  l'utilisateur la fournit dans le `NodeField` de **chargement**, à la position
  `(multiplier_node, imposed_value)` — c'est-à-dire au slot `imposed_<v>` du
  nœud-multiplicateur.
- Le **multiplicateur** se retrouve dans la solution sous le nom `multiplier`
  (`lambda_<v>`) au nœud-multiplicateur ; sa valeur **est** la force de
  réaction de la contrainte.

Les nœuds-multiplicateurs vivent tant que leur maillage **ou** le SubModel les
référence (refcounts) ; quand les deux disparaissent, ils deviennent
collectables. Le SubModel ne décrémente que ce qu'il partage — il n'a rien créé.

## Exemple : Poisson 1-D `-u'' = 0`, `u(0)=0`, `u(1)=1`

Solution analytique `u(x) = x`, multiplicateurs aux bords = flux `±1`. On
compose la conduction thermique avec deux contraintes Dirichlet par l'union `|`
(cf. [Modèle physique](../model.md)) :

```python
import pyrucast

# 1) Maillage + FE space
c = pyrucast.Coords(dim=1)
nodes = [c.add_node([i / 4.0]) for i in range(5)]
mesh = pyrucast.Mesh(c, "SEG2")
for i in range(4):
    mesh.unit().add_cell([nodes[i], nodes[i + 1]])
fes = pyrucast.FiniteElementSpace(mesh)

# 2) Supports de multiplicateurs : barycenter colocalise des nœuds neufs.
imposed_left = pyrucast.poi1_from_nodes([nodes[0]])
imposed_right = pyrucast.poi1_from_nodes([nodes[-1]])
mult_mesh_left = pyrucast.barycenter(imposed_left)
mult_mesh_right = pyrucast.barycenter(imposed_right)
left = pyrucast.Model.dirichlet("T", "q", imposed_left, mult_mesh_left)
right = pyrucast.Model.dirichlet("T", "q", imposed_right, mult_mesh_right)
mult_left = mult_mesh_left.node(0, 0, 0)
mult_right = mult_mesh_right.node(0, 0, 0)

# 3) Modèle complet : conduction + les deux Dirichlet.
model = pyrucast.Model.heat_conduction(fes) | left | right
materials = pyrucast.material_field(model, [("k", 1.0)])

# 4) Chargement : u_d au slot imposed_T des nœuds-multiplicateurs.
rhs_mesh = pyrucast.Mesh(c, "POI1")
rhs_mesh.unit().add_cell([mult_left])
rhs_mesh.unit().add_cell([mult_right])
rhs = pyrucast.NodeField(rhs_mesh, ["imposed_T"])
rhs[0].set_value(mult_left, "imposed_T", 0.0)
rhs[0].set_value(mult_right, "imposed_T", 1.0)

# 5) Assemblage + résolution.
K = pyrucast.stiffness(model, materials)
solution = pyrucast.solve(K, rhs)
assert abs(solution.value(nodes[2], "T") - 0.5) < 1e-10          # T au milieu
assert abs(solution.value(mult_left, "lambda_T") - 1.0) < 1e-10  # flux à gauche
```

La forme Rust équivalente (constructeurs au niveau parent, composés par
`union`) est dans le chapitre [Modèle physique](../model.md).

## Limitations actuelles

- `imposed_mesh` et `multiplier_mesh` sont des maillages **POI1** (contrainte
  par nœud). Les contraintes réparties (sur une arête entière) passeront par un
  bloc `C` issu d'une intégration, comme [`flux`](../operateurs/assemblage.md) pour
  les seconds membres.
- Seule la valeur imposée **constante par nœud** est gérée ; une valeur
  spatialement variable se fournit nœud par nœud dans le chargement.
