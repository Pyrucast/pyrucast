# Python & conventions pyrucast

Cast3M invente un langage de commande dédié, Gibiane, avec ses propres
règles de syntaxe. pyrucast fait le choix inverse : **Python ordinaire**,
sans surcouche — mais avec des conventions de nommage strictes qui jouent le
même rôle que la grammaire de Gibiane. Les connaître à l'avance évite de
chercher à tâtons dans quel sous-module vit telle fonction.

Détail complet : [Correspondance Rust ↔ Python](../correspondance-rust-python.md).

## Empaquetage

```python
import pyrucast as pc
```

`pyrucast` est un paquet **mixed Rust/Python** (maturin) : l'extension
compilée (tout le calcul) vit dans le sous-module privé `pyrucast._pyrucast`
; le paquet public la ré-exporte et y ajoute une petite couche Python pure
de plus haut niveau (`pyrucast.thermomechanics`). À l'usage, aucune
distinction n'est visible :

```python
pc.assemble.stiffness(...)            # opérateur Rust (extension compilée)
pc.thermomechanics.step_by_step(...)  # fonction Python pure
```

## Les objets : classes au niveau racine

Chaque **conteneur** — l'équivalent des « objets » Gibiane (`MAILLAGE`,
`CHPOINT`, `TABLE`…) — est une classe Python au niveau racine du paquet,
**même nom que la structure Rust** :

```python
c = pc.Coords(2)              # Cast3M : OPTI 'DIME' 2
n = c.add_node([0.0, 0.0])    # Cast3M : POIN 0. 0. ;
mesh = pc.Mesh(c, "TRI3")     # Cast3M : MAILLAGE (implicite via un opérateur)
```

Onze conteneurs couvrent tout : `Coords`, `Node`, `Mesh`, `FiniteElementSpace`,
`NodeField`, `ElementField`, `Model`, `Matrix`, `Evolution`, plus leurs vues
`Sub*` (voir plus bas). Contrairement à Gibiane, **pas de typage dynamique
surprise** : `Coords(2)` est toujours un `Coords`, jamais autre chose selon
le contexte.

## Les opérateurs : fonctions rangées par thème

Cast3M lit les quatre premiers caractères d'un nom d'opérateur (`DROITE` ⇔
`DROI`) et laisse tous les opérateurs dans un espace de noms plat. pyrucast
range chaque **verbe** (une fonction libre, l'équivalent d'un opérateur
Gibiane) dans un sous-module nommé d'après son thème — miroir direct de
l'arborescence Rust (`src/ops/<thème>/`) :

| pyrucast (Python) | thème | Cast3M (le plus proche) |
|---|---|---|
| `pc.mesher.line_seg2`, `pc.mesher.surface`, `pc.mesher.sweep_qua4`… | maillage | `DROITE`, `SURF`, `VOLU`, `TRAN` |
| `pc.field.gradient`, `pc.field.select`, `pc.field.mask`… | champs | `GRAD`, `MASQUE` |
| `pc.assemble.stiffness`, `pc.assemble.mass`, `pc.assemble.flux`… | assemblage | `RIGI`, `MASS`, `FLUX`/`PRES` |
| `pc.behavior.integrate_behavior` | comportement | `COMP` |
| `pc.solver.solve`, `pc.solver.solve_unilateral` | solveur | `RESO` |
| `pc.build.material_field` | construction | `MATE` |
| `pc.export.export_vtk` | export | `SORT 'VTK'` |

Aucun nom raccourci ni forme abrégée : contrairement à Gibiane (`DROI` ⇔
`D`), les noms pyrucast sont toujours complets — l'auto-complétion de
l'éditeur remplace l'avantage de la frappe courte.

## Composer, pas boucler : agrégats et union `|`

Sept conteneurs partagent un même protocole d'**agrégat** — la notion la
plus proche de Gibiane `ET` :

```python
modele = pc.Model.heat_conduction(fes) | pc.Model.convection(bord_fes)
modele = modele | pc.Model.dirichlet("T", "q", impose, multiplicateur)
```

`|` unit deux agrégats du même type (`Mesh | Mesh`, `Model | Model`,
`NodeField | NodeField`…) — l'équivalent de `ET` en Gibiane
(`cex = l12 ET c23 ET c34 ...`). `len(agg)`, `agg[i]` (une vue, jamais une
copie), `agg.unit()` (l'unique sous-objet, erreur sinon) complètent le
protocole. **`|` exige des supports disjoints** — deux contributions à la
même valeur d'un même nœud lèvent une erreur explicite plutôt que de se
sommer silencieusement (voir la note correspondante dans
[Calcul thermique](thermique.md)).

L'arithmétique de champs (`+ - * / **`) est réservée aux **valeurs**
(construire un résidu, une charge scalée) — jamais à la composition
d'agrégats. C'est elle qui remplace la plupart des boucles `REPE` de
Gibiane : `residual = f_ext - f_int`, `u = u + du`, sans jamais itérer nœud
par nœud.

## Primal / dual : la convention qui remplace `BLOQ`/`DEPI`

Chaque physique déclare une paire **variable primale / variable duale** par
degré de liberté — `u_x`/`f_x` (déplacement/force), `T`/`q`
(température/flux), `w`/`f_w` (flèche/effort tranchant pour Timoshenko).
Un blocage Dirichlet cible toujours la **variable duale** :

```python
pc.Model.dirichlet("T", "q", impose, multiplicateur)     # Cast3M : BLOQ 'T' ...
pc.Model.dirichlet("u_x", "f_x", impose, multiplicateur) # Cast3M : BLOQ 'UX' ...
```

`model.dual_of("u_x")` renvoie `"f_x"` sans avoir à la mémoriser — utile
pour les contraintes MPC, dont chaque terme cible aussi une variable duale.

## Pas de mode interactif, pas de procédures Gibiane

- Gibiane bascule en mode interactif sur une ligne vide ou `OPTI 'DONN' 5`.
  Python n'a pas cette notion : un script s'exécute jusqu'au bout, ou on
  travaille directement dans un interpréteur/notebook.
- Les procédures Gibiane (`DEBP`/`FINP`) sont, en Python, de simples
  fonctions — aucune syntaxe dédiée à apprendre.
- Pas de commentaire `*` en début de ligne : les commentaires Python
  commencent par `#`, comme dans le reste du langage.

La suite de la formation applique ces conventions sur un cas fil rouge —
direction [Maillage](maillage.md).
