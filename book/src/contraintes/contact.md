# Contact (nœud-surface)

Une contrainte **contact** empêche les nœuds d'un maillage **esclave** de
pénétrer une surface **maître** — le sibling **unilatéral** du
[baignage](embedded.md). Chaque nœud esclave `s` est apparié à sa facette
maître la plus proche (poids de projection `Nᵢ(ξ)`, normale `n`, jeu initial
signé `g₀`), et la non-pénétration linéarisée s'écrit, par nœud esclave :

\\[
g_0 + n \cdot u(s) - \sum_i N_i(\xi)\\, n \cdot u(\text{maître}_i) \ \geq\ 0.
\\]

C'est une **relation unilatérale** (`≥`, cf. la section [Relations
unilatérales](../contraintes.md)) à coefficients variant par nœud (comme le
baignage) et couplant **toutes les composantes** du déplacement via la
normale. Le modèle se résout avec
[`solve`](../operateurs/solveur.md), qui lit l'inégalité dans la matrice ; le multiplicateur `λ ≤ 0`
porte la réaction de contact (`λ = 0` quand la paire est décollée).

Implémentation : `src/models/contact.rs` ; opérateur `ops::model::contact(…)`.

## Appariement à la construction (petits glissements)

L'appariement est calculé **une seule fois, à la construction**, par
**projection au point le plus proche** de chaque nœud esclave sur la surface
maître (`ops::geom::project_points`, cf.
[Opérateurs géométriques](../operateurs/geometrie.md)) : facette, `ξ` (clampé
au domaine de référence — un nœud face à un bord se projette sur le bord),
poids `Nᵢ(ξ)`, normale et jeu signé. Appariement et normale sont ensuite
**figés** : c'est le contact **linéarisé** (petits déplacements, petits
glissements, sans frottement). Les deux maillages doivent partager un même
`Coords`.

**Orientation.** La surface maître doit être **orientée de façon cohérente**,
normale pointant **vers le corps esclave** : en 2D la normale d'un `SEG2` est
la tangente tournée de −90° (`n = (t_y, −t_x)`), en 3D celle d'un `TRI3`/`QUA4`
suit la règle de la main droite sur l'ordre des nœuds. Le jeu `g₀` est alors
positif quand c'est décollé, négatif quand ça pénètre.

Comme toute contrainte, il crée ses **nœuds-multiplicateurs**, un par
relation (donc par nœud esclave), par [`barycenter`](../operateurs/maillage.md)
sur le maillage esclave ; on les retrouve avec `Model.multiplier_mesh()`.

## Une relation par nœud esclave

Les noms sont ceux de **toute** relation de contrainte :

| rôle | nom |
|---|---|
| primale propre = réaction de contact `λ` | `lambda` |
| duale propre = ligne de contrainte + slot de `−g₀` | `imposed` |

Signature complète :

```text
model.contact(slave, master, target, variables)
# variables : les primales de déplacement de la cible, une PAR dimension
#             d'espace, dans l'ordre ambiant, p.ex. ["u_x", "u_y"]
```

Contrairement au baignage (une relation *par composante*), le contact écrit
**une seule relation scalaire** par nœud esclave : la normale couple les
composantes entre elles (coefficients `+n_c` sur l'esclave, `−Nᵢ·n_c` sur
chaque nœud maître). `variables` doit donc en donner exactement une par
dimension.

## Second membre : le jeu, porté par le contact

Le second membre de chaque relation est `−g₀` — une donnée **géométrique**,
que le sous-modèle connaît déjà et que l'utilisateur n'a pas à choisir. Le
contact la **porte** donc comme une charge : [`external_forces`](../operateurs/comportement.md)
la rend, au slot `imposed` de chaque nœud multiplicateur, avec les autres
charges du modèle. Personne ne la calcule à la main, personne ne peut
l'oublier — l'oubli se lisait comme un jeu nul, deux blocs écartés collés à
distance.

```python
{{#include ../../../tests/python/test_doc_contraintes.py:contact_rhs}}
```

## Grands glissements : mettre le contact à jour

L'appariement figé ne vaut que tant que le point projeté ne glisse pas :
quand la facette change ou que la normale tourne, la relation écrite à la
construction n'est plus le jeu. `update_contact(model, displacement)` (aussi
`Model.update_contact(displacement)`) rend le modèle avec ses contacts
**réappariés** sur les positions `X + u` — `X` la configuration active, `u` lu
sur les variables du contact —, les autres sous-modèles partagés tels quels.

Concrètement, pour chaque nœud esclave `s` : la facette maître la plus proche
de `x_s = X_s + u_s`, le point projeté (donc `ξ` et les poids `Nᵢ`) et la
normale `n` sont recalculés **sur les positions déplacées** ; le nœud
multiplicateur, lui, ne change pas. La ligne de la matrice reste la même
(`(multiplicateur, imposed)`, l'inconnue `lambda` aussi) ; ses coefficients
(`+n` sur l'esclave, `−Nᵢ·n` sur les nœuds de la nouvelle facette) et son
second membre changent.

**Le jeu rangé.** Le solveur écrit toute relation en déplacement total depuis
`X`. Mis à jour pour un déplacement `u_k`, le contact range donc le jeu du
nouvel appariement mesuré **sur `X`** :

\\[
\tilde g = n \cdot \big(X_s - \textstyle\sum_i N_i X_{m_i}\big),
\qquad \tilde g + C\\,u = g(X+u_k) + C\\,(u - u_k),
\\]

soit le jeu réel de `X + u_k`, linéarisé là.

Exemple : un nœud parti de `Y = 1` au-dessus d'un sol `y = 0`, descendu de
`0,7` : son jeu actuel vaut `0,3`, `g̃` vaut `1`, et la relation reste
`u_y ≥ −1` — il ne peut pas descendre de plus de 1 depuis sa position de
référence. Le contact mis à jour porte ce `−g̃` comme second membre ;
`update_contact(model, 0)` redonne la construction.

**Des coefficients exacts.** Pour une projection au point le plus proche, la
variation du jeu se réduit à `n·(δx_s − Σ Nᵢ δx_mᵢ)` : le glissement du point
projeté est tangent à la facette, et la normale reste unitaire. Seul le terme
du second ordre — la raideur géométrique du contact — est négligé, comme dans
Cast3M.

```python
{{#include ../../../tests/python/test_doc_contraintes.py:update_contact}}
```

Le [pas à pas](../pas-a-pas.md#grands-déplacements) s'en sert en grands
déplacements : il met ses contacts à jour chaque fois qu'il reconstruit son
opérateur.

## Exemple : patch test à deux blocs

Deux blocs élastiques empilés (jeu initial `g₀`), `u_x` bloqué partout
(colonne uniaxiale), pression `S` sur le bloc du haut : le contact se ferme et
transmet exactement `σ_yy = −S` ; les réactions `−λᵢ` sont les forces nodales
cohérentes de la pression (`Σ(−λᵢ) = S`). En soulevant le bloc du haut, toutes
les paires se relâchent (`λ = 0` exactement).

```python
{{#include ../../../tests/python/test_doc_contraintes.py:contact}}
```

Le déroulé complet (2D et 3D) est dans `tests/contact.rs` et
`tests/python/test_contact.py`.

## Périmètre v1 et suites

- **Appariement à la demande** : figé à la construction, renouvelé par
  `update_contact` ; le pas à pas le fait de lui-même en grands déplacements.
- **Sans frottement** : seul le jeu normal est contraint, le glissement
  tangentiel est libre.
- Nœud-surface **simple passe** : pas de traitement maître/esclave symétrique,
  pas de mortar.
