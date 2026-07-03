# Multi-points (MPC)

Une **contrainte multi-points** (MPC) impose une **relation linéaire** entre
degrés de liberté :

\\[
\sum_k a_k \, u(n_k, v_k) = g .
\\]

C'est une [contrainte](../contraintes.md) imposée par **multiplicateurs de
Lagrange**, exactement comme [Dirichlet](dirichlet.md) — dont elle est la
**généralisation** : Dirichlet est la relation à un seul terme `1·u = u_d`
(coefficient 1), une MPC en a autant qu'on veut, avec des coefficients
quelconques. Aucun matériau, aucune loi de comportement ; elle ne crée aucun
nœud et ne mute jamais le `Coords`.

Implémentation : `src/models/mpc.rs` ; constructeur `Model::mpc(…)`.

## Mise en donnée : un maillage par terme

Chaque **terme** est un tuple `(maillage POI1, variable, dual, coefficient)`.
Tous les terme-maillages et le `multiplier_mesh` sont **appariés
élément-par-élément** : la relation `r` relie la `r`-ème cellule de *chaque*
terme-maillage au `r`-ème nœud multiplicateur. Une périodicité entre deux
surfaces de `N` nœuds est donc `N` relations d'un coup, **vectorisées** sur les
cellules — sans boucle Python.

> **Contrat d'appariement.** L'ordre cohérent des maillages appariés est à la
> charge de l'utilisateur : la cellule `r` de chaque terme-maillage doit
> désigner des nœuds **partenaires** (p. ex. le nœud d'entrée et son image
> périodique). Le modèle vérifie seulement que tout est POI1, partage un même
> `Coords`, et a le même nombre de cellules par sous-maillage.

Le `dual` de chaque terme (la ligne où atterrit la réaction `aₖ·λ`) se trouve
facilement avec **`Model.dual_of(variable)`** — appariement positionnel
`primal_vars[i] ↔ dual_vars[i]` de la physique qui déclare la variable
(`"T" → "q"`, `"u_x" → "f_x"`, …).

## Deux noms de variables

L'MPC partage **une** paire de variables entre toutes ses relations, toutes deux
**surchargeables** :

| rôle | nom | défaut |
|---|---|---|
| primale propre = multiplicateur `λ` (inconnue du système) | `multiplier` | `lambda_mpc` |
| duale propre = ligne de contrainte + **slot** où l'utilisateur écrit `g` | `imposed_value` | `mpc_rhs` |

Signature complète :

```text
Model.mpc(terms, multiplier_mesh, multiplier=None, imposed_value=None)
# terms : liste de (mesh, variable, dual, coefficient)
```

## Les blocs `C` / `Cᵀ`

À l'assemblage, l'MPC contribue **une paire de blocs par (sous-maillage, terme)**,
via le même constructeur partagé que Dirichlet mais avec le coefficient `aₖ` au
lieu de `1` (chaque bloc est marqué non-symétrique ; seule l'union `C ∪ Cᵀ`
l'est — propriété **globale** du système point-selle) :

- **bloc C** : `(multiplier_node, imposed_value) × (nœud_k, variable_k) = aₖ`
- **bloc Cᵀ** : `(nœud_k, dual_k) × (multiplier_node, multiplier) = aₖ`

Tous les termes d'une relation partagent le **même nœud multiplicateur** et la
**même ligne `imposed_value`** : c'est ce qui les additionne dans une seule
équation `Σₖ aₖ uₖ = g`.

## Second membre et réaction

- Le second membre `g` n'est **pas** stocké dans le SubModel : l'utilisateur
  l'écrit dans le `NodeField` de **chargement**, au slot `mpc_rhs` du
  nœud-multiplicateur (défaut `g = 0` — le cas homogène des égalités et
  périodicités).
- Le **multiplicateur** se retrouve dans la solution sous `lambda_mpc` au
  nœud-multiplicateur ; sa valeur **est** la force de réaction de la contrainte.

## Exemple : relation `T(1) − T(0) = 1`

Sur la barre 1-D `-u'' = 0`, un Dirichlet `T(0) = 0` et une MPC à deux termes
`1·T(1) − 1·T(0) = 1` imposent `T(1) = 1`, d'où la solution linéaire `u(x) = x` :

```python
import pyrucast

c = pyrucast.Coords(dim=1)
nodes = [c.add_node([i / 4.0]) for i in range(5)]
mesh = pyrucast.Mesh(c, "SEG2")
for i in range(4):
    mesh.unit().add_cell([nodes[i], nodes[i + 1]])
fes = pyrucast.FiniteElementSpace(mesh)

base = pyrucast.Model.heat_conduction(fes)
dual = base.dual_of("T")  # "q"

# Dirichlet T(0) = 0.
imposed0 = pyrucast.poi1_from_nodes([nodes[0]])
mult0 = pyrucast.barycenter(imposed0)
dirichlet = pyrucast.Model.dirichlet("T", dual, imposed0, mult0)

# MPC 1·T(dernier) − 1·T(0) = 1.
mesh_last = pyrucast.poi1_from_nodes([nodes[-1]])
mesh_first = pyrucast.poi1_from_nodes([nodes[0]])
mult_mpc = pyrucast.barycenter(mesh_last)
mpc = pyrucast.Model.mpc(
    [(mesh_last, "T", dual, 1.0), (mesh_first, "T", dual, -1.0)],
    mult_mpc,
)

model = base | dirichlet | mpc
materials = pyrucast.material_field(model, [("k", 1.0)])

# Chargement : valeur imposée de Dirichlet + second membre g de la MPC. Le
# helper `constraint_rhs` désigne chaque relation par un nœud (nœud contraint
# pour Dirichlet, nœud-terme pour la MPC) et retrouve seul le nœud-multiplicateur
# et la composante (`imposed_T`, `mpc_rhs`). On fusionne les deux avec `|`.
rhs = dirichlet.constraint_rhs([(nodes[0], 0.0)]) | mpc.constraint_rhs(
    [(nodes[-1], 1.0)]
)

solution = pyrucast.solve(pyrucast.stiffness(model, materials), rhs)
assert abs(solution.value(nodes[2], "T") - 0.5) < 1e-10
```

L'exemple complet est dans `examples/mpc_periodicite.py`.

## Limitations actuelles

- Les terme-maillages sont **POI1** (un nœud par relation) ; l'appariement est
  positionnel (aucun outil géométrique fourni).
- Les coefficients sont des **scalaires** par terme (constants sur les cellules
  du terme-maillage) ; les coefficients variant par nœud (bras de levier d'une
  liaison rigide) restent à faire.
- Une **seule** relation reliant un grand nombre de termes (`Σ` sur 100 nœuds)
  demanderait autant de terme-maillages ; ce cas passera par une extension
  « cellule multi-nœuds ». Beaucoup de relations **parallèles** (le cas courant)
  sont déjà vectorisées.
