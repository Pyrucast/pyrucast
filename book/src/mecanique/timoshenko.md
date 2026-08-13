# Poutre de Timoshenko

Poutre **déformable en cisaillement** : élément `SEG2`, dans la configuration
que lui donne la dimension du maillage. C'est **une seule physique** — elle
remplace les anciens `Model.frame` (portique plan) et `Model.frame3d` (cadre
spatial), qui en étaient les cas 2-D et 3-D.

| `Coords` | DDL par nœud | matériau | efforts de section |
|---|---|---|---|
| 1-D | `w`, `theta` | `E, I, G, A_s` | `M, V` |
| 2-D | `u_x, u_y, r_z` | `+ A` | `N, M, V` |
| 3-D | six | `E, A, I_y, I_z, J, G, A_sy, A_sz` | `N, M_y, M_z, T, V_y, V_z` |

Il n'y a rien à choisir : tout ce qui distingue les trois — nombre et noms des
DDL, jeu matériau, efforts rendus, présence d'un terme axial, d'une torsion,
d'une rotation vers les axes globaux — découle de la dimension. On relit les
noms obtenus par `model.primal_vars()`.

## Équations continues résolues

La section reste plane mais **non normale** à l'axe déformé : la rotation `θ`
est un champ indépendant, et la distorsion `γ = w' − θ` une déformation à part
entière. C'est toute la différence avec
[Euler-Bernoulli](bernoulli.md), où `θ = w'` et où `γ` n'existe pas.

- **cinématique** : courbure \\( \kappa = \theta' \\), distorsion \\( \gamma = w' - \theta \\) ;
- **efforts** : \\( M = EI\,\theta' \\), \\( V = G A_s (w' - \theta) \\) ;
- **équilibre** : \\( V' + q = 0 \\), \\( M' - V = 0 \\).

Ces **deux** équations sont du second ordre, là où Bernoulli en a une seule du
quatrième. C'est la contrepartie de l'hypothèse cinématique : libérer `θ` de
`w'` abaisse l'ordre de l'équation, et abaisse avec lui l'exigence de continuité
— le C⁰ suffit là où Bernoulli réclame du C¹.

## Forme discrétisée — l'élément exact

L'élément assemblé est la **solution exacte** de ces deux équations sur une
travée libre d'efforts répartis. Ses fonctions de forme sont cubiques en `w` et
quadratiques en `θ`, et elles portent le matériau par

\\[
\Phi = \frac{12\,E I}{G A_s L^2},
\\]

le rapport des souplesses de flexion et de cisaillement. La flexion s'écrit
alors en forme fermée :

\\[
K_b = \frac{EI}{L^3(1+\Phi)}
\begin{bmatrix}
 12 & 6L & -12 & 6L \\\\
 6L & (4+\Phi)L^2 & -6L & (2-\Phi)L^2 \\\\
-12 & -6L & 12 & -6L \\\\
 6L & (2-\Phi)L^2 & -6L & (4+\Phi)L^2
\end{bmatrix}.
\\]

L'élément est **exact aux nœuds** pour des charges d'extrémité : un élément par
barre suffit. On lit directement sur cette matrice que la raideur en flèche est
la combinaison **en série** des deux souplesses,

\\[
K_{ww} = \frac{12EI}{L^3(1+\Phi)}
       = \frac{1}{\dfrac{L^3}{12EI} + \dfrac{L}{G A_s}},
\\]

— on fléchit le tronçon *et* on le cisaille, les deux cèdent l'un après
l'autre. Et \\( \Phi = 0 \\) redonne terme pour terme la matrice
d'Euler-Bernoulli : « Bernoulli est la limite sans cisaillement » est une
propriété vérifiée par un test, pas une phrase.

### L'espace EF ne porte aucune base

Ces fonctions de forme dépendent du **matériau** par \\( \Phi \\). Aucun espace
éléments finis ne peut donc les tabuler — il tabule par type d'élément, pas par
maille. L'espace déclare en conséquence
[`MODEL_EMBEDDED`](../fe-space.md#pas-de-base-du-tout--model_embedded) : la
formulation possède son interpolation, et le dit.

```python
fes = pyrucast.FiniteElementSpace(maillage, interpolation="MODEL_EMBEDDED")
poutre = pyrucast.Model.timoshenko(fes)
```

> **Ce que remplace cet élément.** La version précédente était **linéaire**, à
> cisaillement sous-intégré : elle convergeait au raffinement au lieu d'être
> exacte, et déclarait une interpolation de Lagrange qu'elle utilisait
> réellement. Le portique 2-D était dans ce cas, le cadre 3-D employait déjà la
> forme exacte — deux modèles frères, deux théories discrètes. Ils n'en font
> plus qu'une.

## Variables et matériau

Voir le tableau d'ouverture. `rho` est **facultatif**, exigé par la seule
matrice de masse ; en configuration 1-D l'aire pleine `A` l'est aussi, la
rigidité n'utilisant que l'aire de cisaillement.

Le comportement (`COMP`) rend les efforts de section par une loi **linéaire**,
à partir des déformations généralisées produites par
[`beam_deformation`](../operateurs/champs.md), un opérateur pour les trois configurations.

## Mise en donnée (Rust, testé)

Console encastrée, charge transverse `P` au bout libre ; solution analytique
`w = P·L³/(3EI) + P·L/(G·A_s)` — les deux souplesses, en série.

```rust,ignore
{{#include ../../../tests/timoshenko.rs:example}}
```

Le portique plan, où l'axial et la flexion se découplent :

```rust,ignore
{{#include ../../../tests/frame.rs:example}}
```

Et le cadre spatial, avec sa torsion :

```rust,ignore
{{#include ../../../tests/frame3d.rs:example}}
```

## Exemple Python

```python
{{#include ../../../examples/timoshenko.py}}
```

## Compléments

**Masse et rigidité géométrique.** La masse cohérente assemblée est celle de
l'élément **linéaire**, pas de l'élément exact — pratique d'ingénierie usuelle,
et ce que cette physique a toujours assemblé. C'est une incohérence avec la
rigidité, signalée plutôt que cachée, et le prochain point à traiter. La
rigidité géométrique demande un effort axial pour raidir la barre : la
configuration 1-D, en flexion pure, n'en déclare donc aucune.

**Reconstruction des efforts.** `beam_deformation` rend des déformations
**élément-constantes**, héritées de l'élément linéaire.
Avec l'élément exact, la courbure varie dans la maille : la reconstruction est
donc une approximation, assumée par la formulation. Le repère local, lui, est
déduit automatiquement de la géométrie (référence globale Z, ou Y pour une barre
verticale), ce qui convient aux sections symétriques.

**Le repère de rotation.** Le portique plan nommait sa rotation `rz` quand tout
le reste du dépôt écrivait `r_z`. La fusion l'a fait sortir immédiatement — une
matrice non carrée au solveur — et c'est `r_z` qui l'emporte.
