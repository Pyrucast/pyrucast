# Contact (nœud-surface)

Une contrainte **contact** empêche les nœuds d'un maillage **esclave** de
pénétrer une surface **maître** — le sibling **unilatéral** du
[baignage](embedded.md). Chaque nœud esclave `s` est apparié à sa facette
maître la plus proche (poids de projection `Nᵢ(ξ)`, normale `n`, jeu initial
signé `g₀`), et la non-pénétration linéarisée s'écrit, par nœud esclave :

\\[
g_0 + n \cdot u(s) - \sum_i N_i(\xi)\, n \cdot u(\text{maître}_i) \ \geq\ 0.
\\]

C'est une **relation unilatérale** (`≥`, cf. la section [Relations
unilatérales](../contraintes.md)) à coefficients variant par nœud (comme le
baignage) et couplant **toutes les composantes** du déplacement via la
normale. Le modèle se résout avec
[`solve_unilateral`](../operateurs/solveur.md) ; le multiplicateur `λ ≤ 0`
porte la réaction de contact (`λ = 0` quand la paire est décollée).

Implémentation : `src/models/contact.rs` ; constructeur `Model::contact(…)`.

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

Les **nœuds-multiplicateurs** sont mintés en interne (un par nœud esclave,
colocalisé), accessibles après coup avec `Model.multiplier_mesh()`.

## Une relation par nœud esclave

Toutes les relations partagent la paire de variables du sous-modèle
(surchargeables) :

| rôle | nom | défaut |
|---|---|---|
| primale propre = réaction de contact `λ` | `multiplier` | `lambda_contact` |
| duale propre = ligne de contrainte + slot de `−g₀` | `imposed_value` | `contact_gap` |

Signature complète :

```text
Model.contact(slave, master, components, multiplier=None, imposed_value=None)
# components : une paire (variable, target_dual) PAR dimension d'espace,
#              dans l'ordre ambiant, p.ex. [("u_x","f_x"), ("u_y","f_y")]
```

Contrairement au baignage (une relation *par composante*), le contact écrit
**une seule relation scalaire** par nœud esclave : la normale couple les
composantes entre elles (coefficients `+n_c` sur l'esclave, `−Nᵢ·n_c` sur
chaque nœud maître). `components` doit donc en donner exactement une par
dimension.

## Second membre : le helper `contact_gaps`

Le second membre de chaque relation est `−g₀` — une donnée **géométrique**
que le sous-modèle connaît déjà. Le helper la transforme en champ de
chargement, à fusionner avec `|` :

```python
rhs = traction | model.contact_gaps()
```

L'omettre revient à traiter toutes les paires comme **initialement en
contact** (`g₀ = 0`).

## Exemple : patch test à deux blocs

Deux blocs élastiques empilés (jeu initial `g₀`), `u_x` bloqué partout
(colonne uniaxiale), pression `S` sur le bloc du haut : le contact se ferme et
transmet exactement `σ_yy = −S` ; les réactions `−λᵢ` sont les forces nodales
cohérentes de la pression (`Σ(−λᵢ) = S`). En soulevant le bloc du haut, toutes
les paires se relâchent (`λ = 0` exactement).

```python
# Maître : bord supérieur du bloc bas, parcouru en −x (normale +y, vers l'esclave).
master = pyrucast.Mesh(c, "SEG2")
for i in reversed(range(N)):
    master.unit().add_cell([bottom[idx(i + 1, N)], bottom[idx(i, N)]])
# Esclave : nœuds du bord inférieur du bloc haut.
slave = pyrucast.poi1_from_nodes([top[idx(i, 0)] for i in range(N + 1)])

contact = pyrucast.Model.contact(slave, master, [("u_x", "f_x"), ("u_y", "f_y")])
model = elasticite | appuis | contact

rhs = pyrucast.flux(edge_fes[0], -S, "f_y") | model.contact_gaps()
solution = pyrucast.solve_unilateral(model, pyrucast.stiffness(model, materials), rhs)

reaction = solution.value(mult_node, "lambda_contact")   # ≤ 0 collé, 0 décollé
```

Le déroulé complet (2D et 3D) est dans `tests/contact.rs` et
`tests/python/test_contact.py`.

## Périmètre v1 et suites

- **Petits glissements** : appariement et normale figés à la construction ;
  un grand glissement demandera un ré-appariement en boucle (orchestré côté
  Python, comme le Newton de la plasticité).
- **Sans frottement** : seul le jeu normal est contraint, le glissement
  tangentiel est libre.
- Nœud-surface **simple passe** : pas de traitement maître/esclave symétrique,
  pas de mortar.
