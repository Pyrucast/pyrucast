# Baignage (embedded)

Une contrainte **embedded** (ou « baignage ») lie le champ de **chaque nœud
d'un maillage immergé** à l'**interpolation d'un maillage hôte** au même point de
l'espace. Pour un nœud immergé `p` logé dans la maille hôte de fonctions de forme
`Nᵢ(ξ)`, et pour chaque composante `c` :

\\[
u_c(p) - \sum_i N_i(\xi_p)\\, u_c(\text{hôte}_i) = g_c
\qquad (g_c = 0 : \text{liaison rigide}).
\\]

C'est l'archétype d'une **barre baignée dans un volume** : les nœuds de la barre
suivent le champ de déplacement volumique, **sans** que les deux maillages
partagent de nœud. Comme [Dirichlet](dirichlet.md) et [MPC](mpc.md), c'est une
[contrainte](../contraintes.md) par **multiplicateurs de Lagrange** : ni
matériau, ni loi de comportement.

C'est aussi la réponse au cas laissé ouvert par la MPC : des **coefficients qui
varient par nœud** (ici les `Nᵢ(ξ_p)`, différents à chaque nœud immergé), qu'une
`model.mpc` à coefficients scalaires ne sait pas exprimer de façon compacte.

Implémentation : `src/models/embedded.rs` ; opérateur `ops::model::embedded(…)`.

## Localisation à la construction

Les poids de couplage `Nᵢ(ξ_p)` sont calculés **une seule fois, à la
construction**, en localisant chaque nœud immergé dans le maillage hôte
(mapping iso-paramétrique inverse : un Newton sur le résidu
`x − Σ Nᵢ(ξ)·Xᵢ`, avec test d'appartenance au domaine de référence de la maille).
Un nœud immergé qui **ne tombe dans aucune maille hôte** est une **erreur** : le
maillage immergé doit être contenu dans l'hôte. Les deux maillages doivent
partager un même `Coords` (les identifiants de nœuds y sont relatifs).

Comme toute contrainte, elle crée ses **nœuds-multiplicateurs**, un par
relation, par [`barycenter`](../operateurs/maillage.md) sur le maillage
immergé. On y accède après coup avec **`Model.multiplier_mesh()`**, qui porte
un sous-maillage POI1 par composante.

## Une relation par (nœud immergé × composante)

Chaque composante a **ses** nœuds-multiplicateurs, colocalisés avec les nœuds
immergés :

| rôle | nom | fourniture |
|---|---|---|
| primale contrainte (colonne partagée immergé ↔ hôte) | `variable` | requis |
| duale cible où atterrit la réaction | lue dans `target` | déduit |
| primale propre = multiplicateur `λ` | `lambda` | commun à toute relation |
| duale propre = ligne de contrainte + slot de `g` | `imposed` | commun à toute relation |

Signature complète :

```text
model.embedded(immersed, host, target, variables, tol=1e-6)
# variables : les primales à lier, p.ex. ["u_x", "u_y", "u_z"]
```

La **cible** est le modèle contraint : chaque variable doit être une de ses
primales — refusé sinon, en nommant ce qu'elle déclare — et la ligne duale où
atterrit la réaction s'y lit. On ne redonne donc que les noms de variables,
jamais les couples `(primale, duale)`.

## Les blocs `C` / `Cᵀ`

À l'assemblage, la contrainte contribue **une paire de blocs par composante**. Le
nœud immergé porte le coefficient `+1`, chaque nœud hôte son poids `−Nᵢ`, si bien
que chaque relation lit `u_c(p) − Σᵢ Nᵢ·u_c(hôteᵢ) = g_c` :

- **bloc C** : `(multiplier_node, imposed) × (nœud, variable)` = `+1` sur le
  nœud immergé, `−Nᵢ` sur chaque nœud hôte ;
- **bloc Cᵀ** : `(nœud, target_dual) × (multiplier_node, lambda)`, mêmes
  coefficients (réaction réinjectée dans la physique).

## Second membre et réaction

- Le second membre `g` (défaut `0`, la liaison rigide) s'écrit dans le
  `NodeField` de chargement, au slot `imposed` du nœud-multiplicateur de la
  composante.
- Le multiplicateur `lambda` au nœud-multiplicateur **est** la force de
  liaison.

## Exemple : barre baignée dans un HEX8

Un cube HEX8 en conduction thermique, ses huit coins fixés à un champ linéaire
`T(x) = 1 + 2x + 3y + 4z` (que l'interpolation trilinéaire reproduit exactement à
l'intérieur), et un nœud immergé au cœur : sa température résolue **égale**
l'interpolation de l'hôte.

```python
{{#include ../../../tests/python/test_doc_contraintes.py:embedded_complet}}
```

L'exemple complet est dans `examples/barre_baignee.py`. La variante
**vectorielle** — une barre suivant les *déplacements* d'un volume élastique en
`u_x`/`u_y`/`u_z`, le cas qui motive le baignage — est dans
`examples/barre_baignee_elastique.py`.

## Limitations actuelles

- Le maillage immergé est réduit à ses **nœuds** (support POI1 interne) ; on lie
  des nœuds à une interpolation, pas des mailles à des mailles (pas de couplage
  surfacique / cohésif).
- La localisation fait un **balayage** des mailles hôtes (rejet par boîte
  englobante) ; pas encore d'index spatial — coûteux pour de très gros hôtes.
- Types hôtes supportés : tous les éléments à cadre de référence (SEG, TRI, QUA,
  TET, PENTA, HEX — linéaires **et** quadratiques). Le `POI1` n'a pas d'intérieur
  et est ignoré comme hôte.
