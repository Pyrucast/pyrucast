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
matériau, ni loi de comportement ; elle ne mute jamais le `Coords`.

C'est aussi la réponse au cas laissé ouvert par la MPC : des **coefficients qui
varient par nœud** (ici les `Nᵢ(ξ_p)`, différents à chaque nœud immergé), qu'une
`Model.mpc` à coefficients scalaires ne sait pas exprimer de façon compacte.

Implémentation : `src/models/embedded.rs` ; constructeur `Model::embedded(…)`.

## Localisation à la construction

Les poids de couplage `Nᵢ(ξ_p)` sont calculés **une seule fois, à la
construction**, en localisant chaque nœud immergé dans le maillage hôte
(mapping iso-paramétrique inverse : un Newton sur le résidu
`x − Σ Nᵢ(ξ)·Xᵢ`, avec test d'appartenance au domaine de référence de la maille).
Un nœud immergé qui **ne tombe dans aucune maille hôte** est une **erreur** : le
maillage immergé doit être contenu dans l'hôte. Les deux maillages doivent
partager un même `Coords` (les identifiants de nœuds y sont relatifs).

Les **nœuds-multiplicateurs** sont mintés en interne (un par nœud immergé,
colocalisé), contrairement à Dirichlet/MPC où l'utilisateur fournit le
`multiplier_mesh`. On y accède après coup avec **`Model.multiplier_mesh()`**.

## Une relation par (nœud immergé × composante)

Toutes les composantes partagent **un** nœud-multiplicateur par nœud immergé,
chacune portant sa propre paire de variables (toutes **surchargeables**) :

| rôle | nom | défaut |
|---|---|---|
| primale contrainte (colonne partagée immergé ↔ hôte) | `variable` | — (fournie) |
| duale cible où atterrit la réaction | `target_dual` | — (fournie, cf. `dual_of`) |
| primale propre = multiplicateur `λ` | `multiplier` | `lambda_<variable>` |
| duale propre = ligne de contrainte + slot de `g` | `imposed_value` | `imposed_<variable>` |

Signature complète :

```text
Model.embedded(immersed, host, components,
               multipliers=None, imposed_values=None, tol=None)
# components : liste de (variable, target_dual), p.ex. [("u_x","f_x"), ("u_y","f_y")]
```

Chaque `target_dual` se trouve avec **`Model.dual_of(variable)`**
(`"T" → "q"`, `"u_x" → "f_x"`, …).

## Les blocs `C` / `Cᵀ`

À l'assemblage, la contrainte contribue **une paire de blocs par composante**. Le
nœud immergé porte le coefficient `+1`, chaque nœud hôte son poids `−Nᵢ`, si bien
que chaque relation lit `u_c(p) − Σᵢ Nᵢ·u_c(hôteᵢ) = g_c` :

- **bloc C** : `(multiplier_node, imposed_value) × (nœud, variable)` = `+1` sur le
  nœud immergé, `−Nᵢ` sur chaque nœud hôte ;
- **bloc Cᵀ** : `(nœud, target_dual) × (multiplier_node, multiplier)`, mêmes
  coefficients (réaction réinjectée dans la physique).

## Second membre et réaction

- Le second membre `g` (défaut `0`, la liaison rigide) s'écrit dans le
  `NodeField` de chargement, au slot `imposed_<variable>` du nœud-multiplicateur.
- Le multiplicateur `lambda_<variable>` au nœud-multiplicateur **est** la force de
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
