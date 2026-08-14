# Espace éléments finis (`FiniteElementSpace`)

Ce chapitre couvre `FiniteElementSpace` et son objet associé `SubFiniteElementSpace` : la couche **éléments finis** posée par-dessus la couche **géométrique** (`Mesh` / `SubMesh`). Le `Mesh` reste purement géométrique ; le `FiniteElementSpace` lui ajoute la **formulation** (fonctions de forme, points de Gauss, dérivées). Cette séparation permet de réutiliser un même maillage avec plusieurs formulations, et — plus tard — d'admettre un déplacement du maillage tout en réévaluant correctement les grandeurs physiques.

## Architecture

Hiérarchie miroir de celle du maillage :

```text
Mesh                                  FiniteElementSpace
├── SubMesh (ElementType)             ├── SubFiniteElementSpace (Interpolation, QuadratureRule)
├── SubMesh                           ├── SubFiniteElementSpace
└── ...                               └── ...
```

- **`SubFiniteElementSpace`** détient un `Handle<SubMesh>`, une `Interpolation` et une `QuadratureRule`. Il porte les **tables de référence** précalculées une fois pour toute (points de Gauss, fonctions de forme et dérivées de référence évaluées à ces points).
- **`FiniteElementSpace`** détient un `Handle<Mesh>` figé et un `Vec<Handle<SubFiniteElementSpace>>` en correspondance **un-pour-un** avec les sous-maillages. La topologie (connectivité, types d'éléments) est figée à la construction.

`POI1` n'est pas un élément fini au sens classique : un sous-maillage `POI1` est rejeté à la construction d'un `SubFiniteElementSpace`.

> Le catalogue détaillé — fonctions de forme, dérivées et quadrature de **chaque** type — est en section [Éléments finis supportés](elements/index.md). Ce chapitre-ci décrit la machinerie **commune** (mapping isoparamétrique, Jacobien, gradients physiques) partagée par tous.

## Conventions de l'élément de référence

Chaque `ElementType` fixe son repère de référence \\( \xi \\) et la numérotation locale de ses nœuds. Ces conventions sont aussi documentées dans le rustdoc de [`ElementType`](https://docs.rs/) et reproduites ici pour référence centrale.

| ElementType | Repère \\( \xi \\) | Numérotation locale (ordre des nœuds) |
|---|---|---|
| `SEG2` | \\( \xi \in [-1, +1] \\) | nœud 0 en \\( \xi = -1 \\), nœud 1 en \\( \xi = +1 \\) |
| `TRI3` | \\( \xi, \eta \in [0, 1] \\), \\( \xi + \eta \le 1 \\) | \\( (0,0), (1,0), (0,1) \\) — CCW |
| `QUA4` | \\( \xi, \eta \in [-1, +1] \\) | \\( (-1,-1), (1,-1), (1,1), (-1,1) \\) — CCW |
| `TET4` | \\( \xi, \eta, \zeta \in [0, 1] \\), \\( \xi + \eta + \zeta \le 1 \\) | \\( (0,0,0), (1,0,0), (0,1,0), (0,0,1) \\) — face 0-1-2 CCW vue depuis nœud 3 |
| `PYRA5` | \\( \zeta \in [0, 1] \\), \\( \xi, \eta \in [-(1-\zeta), +(1-\zeta)] \\) | base carrée CCW vue depuis l'apex (nœuds 0..3 en \\( \zeta = 0 \\)) puis l'apex : \\( (-1,-1,0), (1,-1,0), (1,1,0), (-1,1,0), (0,0,1) \\) |
| `PENTA6` | \\( \xi, \eta \in [0, 1] \\), \\( \xi + \eta \le 1 \\), \\( \zeta \in [0, 1] \\) | triangle inférieur CCW (nœuds 0..2 en \\( \zeta = 0 \\)) puis triangle supérieur CCW (nœuds 3..5 en \\( \zeta = 1 \\)) — extrusion d'un TRI3 |
| `HEX8` | \\( \xi, \eta, \zeta \in [-1, +1] \\) | face inférieure CCW (nœuds 0..3) puis face supérieure CCW (nœuds 4..7) |

Les types **quadratiques** partagent le repère et la numérotation des sommets de leur parent linéaire, puis ajoutent les nœuds de **milieu d'arête** dans l'ordre d'arêtes de la convention VTK (documenté dans le rustdoc d'`ElementType`) :

| ElementType | Parent | Nœuds de milieu d'arête (dans l'ordre), arête \\( (a,b) \\) |
|---|---|---|
| `SEG3` | `SEG2` | nœud 2 sur \\( (0,1) \\) (\\( \xi = 0 \\)) |
| `TRI6` | `TRI3` | 3 sur \\( (0,1), (1,2), (2,0) \\) |
| `QUA8` | `QUA4` | 4 sur \\( (0,1), (1,2), (2,3), (3,0) \\) |
| `QUA9` | `QUA4` | 4 arêtes comme `QUA8`, puis un nœud **central** 8 en \\( (0,0) \\) |
| `TET10` | `TET4` | 6 sur \\( (0,1), (1,2), (2,0), (0,3), (1,3), (2,3) \\) |
| `PENTA15` | `PENTA6` | 9 : bas \\( (0,1),(1,2),(2,0) \\), haut \\( (3,4),(4,5),(5,3) \\), verticales \\( (0,3),(1,4),(2,5) \\) |
| `HEX20` | `HEX8` | 12 : bas \\( (0,1),(1,2),(2,3),(3,0) \\), haut \\( (4,5),(5,6),(6,7),(7,4) \\), verticales \\( (0,4),(1,5),(2,6),(3,7) \\) |
| `HEX27` | `HEX8` | 12 arêtes comme `HEX20`, puis 6 **centres de face** (`x∓`, `y∓`, `z∓`, nœuds 20..25) et un **centre de volume** 26 |

`QUA8`, `HEX20`, `PENTA15` sont **sérendipité** (arêtes seulement) ; `SEG3`, `TRI6`, `TET10`, `QUA9`, `HEX27` sont des Lagrange complets (`QUA9`/`HEX27` = tenseurs `Q2` complets, avec nœuds de face et central).

Ces conventions sont cohérentes avec celles déjà imposées ailleurs dans le code : orientation CCW des triangles produits par `triangulate_surface`, ordre HEX8/PENTA6 utilisé par `extrude` et `sweep_solid`, ordre des nœuds de milieu d'arête aligné sur VTK (export verbatim) et réaligné à la lecture gmsh.

## Théorie : élément isoparamétrique

### Transformation géométrique de référence

Soit un élément avec \\( n \\) nœuds, de coordonnées physiques \\( \mathbf{x}_1, \dots, \mathbf{x}_n \in \mathbb{R}^{d_s} \\) (avec \\( d_s \\) la dimension géométrique de la `Coords`). La **transformation géométrique** \\( \chi : \hat{K} \to K \\) qui envoie l'élément de référence \\( \hat{K} \\) sur l'élément physique \\( K \\) est interpolée par les mêmes fonctions de forme :

\\[
\mathbf{x}(\xi) = \chi(\xi) = \sum_{i=1}^{n} N_i(\xi)\\, \mathbf{x}_i
\\]

C'est l'hypothèse **isoparamétrique** : la géométrie est interpolée exactement comme un champ scalaire. Avec une interpolation Lagrange-1, \\( \chi \\) est affine sur les simplexes (SEG2, TRI3, TET4) et tri-linéaire sur les tenseurs (QUA4, HEX8).

### Fonctions de forme Lagrange-1

Une fonction de forme Lagrange est définie par la **propriété de Kronecker** : \\( N_i(\xi_j) = \delta_{ij} \\) aux nœuds de référence. Les formules explicites sur les cinq types supportés sont :

**SEG2** (\\( \xi \in [-1, +1] \\)) :
\\[
N_1(\xi) = \frac{1 - \xi}{2}, \qquad
N_2(\xi) = \frac{1 + \xi}{2}
\\]

**TRI3** (coordonnées barycentriques sur le simplexe \\( \xi + \eta \le 1 \\)) :
\\[
N_1 = 1 - \xi - \eta, \qquad
N_2 = \xi, \qquad
N_3 = \eta
\\]

**QUA4** (bilinéaire sur \\( [-1, +1]^2 \\)) :
\\[
N_1 = \tfrac{1}{4}(1-\xi)(1-\eta), \quad
N_2 = \tfrac{1}{4}(1+\xi)(1-\eta), \quad
N_3 = \tfrac{1}{4}(1+\xi)(1+\eta), \quad
N_4 = \tfrac{1}{4}(1-\xi)(1+\eta)
\\]

**TET4** (coordonnées barycentriques sur le simplexe \\( \xi + \eta + \zeta \le 1 \\)) :
\\[
N_1 = 1 - \xi - \eta - \zeta, \quad
N_2 = \xi, \quad
N_3 = \eta, \quad
N_4 = \zeta
\\]

**HEX8** (tri-linéaire sur \\( [-1, +1]^3 \\)). Pour le nœud \\( i \\) de coordonnées de référence \\( (\xi_i, \eta_i, \zeta_i) \in \{-1, +1\}^3 \\),
\\[
N_i(\xi, \eta, \zeta) = \tfrac{1}{8}\\,
(1 + \xi_i\\, \xi)\\,
(1 + \eta_i\\, \eta)\\,
(1 + \zeta_i\\, \zeta)
\\]

**PENTA6** (prisme, produit d'un TRI3 par un SEG2 linéaire sur \\( \zeta \in [0, 1] \\)). Avec les coordonnées barycentriques du triangle \\( L_1 = 1 - \xi - \eta \\), \\( L_2 = \xi \\), \\( L_3 = \eta \\),
\\[
N_j(\xi, \eta, \zeta) = L_j\\,(1 - \zeta) \quad (j = 1, 2, 3), \qquad
N_{j+3}(\xi, \eta, \zeta) = L_j\\,\zeta \quad (j = 1, 2, 3)
\\]

Une propriété immédiate, vérifiée par les tests unitaires : \\( \sum_i N_i(\xi) = 1 \\) en tout \\( \xi \\) (partition de l'unité).

### Fonctions de forme quadratiques (Lagrange-2)

L'interpolation `Lagrange2` couvre les six types quadratiques. Pour les Lagrange complets, avec les coordonnées barycentriques \\( L_i \\) :

- **SEG3** : \\( N_0 = \tfrac12\xi(\xi-1),\\; N_1 = \tfrac12\xi(\xi+1),\\; N_2 = 1-\xi^2 \\).
- **TRI6 / TET10** : sommets \\( L_i(2L_i-1) \\), milieux d'arête \\( 4 L_a L_b \\).

Les sérendipité n'ont pas de nœud de face/intérieur :

- **QUA8** : sommet \\( \tfrac14(1+\xi_i\xi)(1+\eta_i\eta)(\xi_i\xi+\eta_i\eta-1) \\) ; milieu \\( \tfrac12(1-\xi^2)(1+\eta_i\eta) \\) ou \\( \tfrac12(1+\xi_i\xi)(1-\eta^2) \\) selon l'arête.
- **QUA9** : produit tensoriel complet \\( N_i(\xi,\eta) = \ell_{a}(\xi)\\,\ell_{b}(\eta) \\) des trois fonctions de Lagrange 1D \\( \ell_{-}(t)=\tfrac12 t(t-1),\\; \ell_0(t)=1-t^2,\\; \ell_{+}(t)=\tfrac12 t(t+1) \\).
- **HEX27** : produit tensoriel 3D \\( N_i(\xi,\eta,\zeta) = \ell_{a}(\xi)\\,\ell_{b}(\eta)\\,\ell_{c}(\zeta) \\) des mêmes fonctions 1D.
- **HEX20** : sommet \\( \tfrac18(1+\xi_i\xi)(1+\eta_i\eta)(1+\zeta_i\zeta)(\xi_i\xi+\eta_i\eta+\zeta_i\zeta-2) \\) ; milieu \\( \tfrac14(1-\xi^2)(1+\eta_i\eta)(1+\zeta_i\zeta) \\) (et permutations).
- **PENTA15** : produit du TRI6 par un facteur quadratique en \\( \zeta \\), avec correction sérendipité aux sommets.

Toutes vérifient Kronecker (\\( N_i(\xi_j)=\delta_{ij} \\)), partition de l'unité, et leurs dérivées analytiques sont recoupées par différences finies dans les tests.

### Fonctions de forme cubiques d'Hermite (Hermite-3)

Les deux familles précédentes sont **C⁰** : le champ est continu d'un élément au
suivant, sa dérivée ne l'est pas. Cela suffit à toute équation du second ordre,
et à rien d'autre. Une poutre d'Euler-Bernoulli obéit à
\\( (EIw'')'' = q \\), du **quatrième** ordre, dont la forme faible exige un
champ **C¹** — flèche *et* pente continues.

C'est ce que fournit `Hermite3`, sur `SEG2` uniquement :

\\[
\begin{aligned}
H_1 &= \tfrac14(2 - 3\xi + \xi^3), &\qquad H_2 &= \tfrac14(1 - \xi - \xi^2 + \xi^3),\\\\
H_3 &= \tfrac14(2 + 3\xi - \xi^3), &\qquad H_4 &= \tfrac14(-1 - \xi + \xi^2 + \xi^3).
\end{aligned}
\\]

La propriété de Kronecker y porte sur **deux** grandeurs à la fois : à chaque
extrémité, une fonction vaut 1 et a une pente nulle, l'autre vaut 0 et a une
pente 1, et les deux fonctions de l'extrémité opposée s'annulent dans les deux.
D'où l'ordre des degrés de liberté, \\( [w_A,\ w'_A,\ w_B,\ w'_B] \\) — une
valeur et une pente par nœud.

#### Deux conséquences structurelles

**Quatre fonctions pour deux nœuds.** Le nombre de fonctions de forme cesse
d'être le nombre de nœuds ; c'est `shape_count()` qui le donne, et tous les
tableaux de référence sont dimensionnés dessus. `nodes_per_cell()` reste ce
qu'il était, et continue de dimensionner la **géométrie**.

**L'élément devient sous-paramétrique.** La géométrie d'un `SEG2` est un segment
droit, interpolée en Lagrange-1, quel que soit le champ qu'il porte. L'espace
tabule donc **deux** bases :

| base | longueur | rôle |
|---|---|---|
| géométrique (`n_at_g`) | `nodes_per_cell` | le jacobien \\( \partial x/\partial\xi \\) |
| de champ (`field_n_at_g`) | `shape_count` | l'inconnue |

Elles coïncident pour toute interpolation de Lagrange, et l'accesseur de champ
se rabat alors sur la géométrique — d'où un coût mémoire nul dans le cas
courant, et aucun consommateur existant à modifier.

#### Dérivées secondes

`Hermite3` est la seule famille à tabuler \\( \partial^2 N_i/\partial\xi^2 \\),
et c'est cohérent : la courbure est une grandeur **primaire** pour un élément
C¹, alors qu'une base de Lagrange n'en a aucun usage. Sur `SEG2` elles sont
linéaires en \\( \xi \\) — donc la courbure **varie** dans l'élément, là où un
`SEG2` de Lagrange en donnerait une identiquement nulle.

Le passage au physique est une simple règle de chaîne, le terme
\\( \partial J/\partial\xi \\) disparaissant puisqu'un segment a un jacobien
constant :

\\[
\frac{\partial^2 N_i}{\partial x^2} = \frac{1}{J^2}\\,
\frac{\partial^2 N_i}{\partial \xi^2}, \qquad J = \frac{L}{2}.
\\]

#### La pente de référence n'est pas la rotation

`H₂` et `H₄` portent une pente \\( \partial w/\partial\xi \\), tandis que le
degré de liberté d'une poutre est \\( \theta = \partial w/\partial x \\). Les
deux diffèrent du jacobien, \\( \partial w/\partial\xi = J\\,\partial w/\partial x \\).

Ce facteur est **délibérément absent** des fonctions de forme : c'est exactement
le passage référence → physique que toute autre base traverse, donc il vit là où
tous les autres jacobiens sont appliqués. Écrite autrement — avec un `L` dans la
fonction de forme — la base cesserait d'être une grandeur de l'élément de
référence, et ne pourrait plus être tabulée une fois par type d'élément.

#### Ce qui le vérifie

La raideur d'Euler-Bernoulli est le seul oracle capable de falsifier cette base :

\\[
K = \int_0^L EI \left(\frac{\partial^2 N}{\partial x^2}\right)^{\\!\top}
\frac{\partial^2 N}{\partial x^2}\\, dx
= \frac{EI}{L^3}
\begin{bmatrix}
 12 & 6L & -12 & 6L \\\\
 6L & 4L^2 & -6L & 2L^2 \\\\
-12 & -6L & 12 & -6L \\\\
 6L & 2L^2 & -6L & 4L^2
\end{bmatrix}.
\\]

`tests/hermite.rs` intègre le membre de gauche depuis la base tabulée et le
compare à la forme fermée classique, à la précision machine. Une erreur dans les
fonctions, dans leurs dérivées secondes ou dans le facteur jacobien des pentes
atterrit dans cette comparaison — ce qui vaut mieux que seize assertions sur la
base seule. L'intégrande étant quadratique en \\( \xi \\), deux points de Gauss
l'intègrent **exactement**.

### Pas de base du tout : `MODEL_EMBEDDED`

Un élément structurel dont la matrice élémentaire est une **forme fermée** —
une barre, un portique — n'évalue jamais une fonction de forme. Son intégrale a
été faite une fois, au crayon, et seul le résultat figure dans le code.

Déclarer une interpolation de Lagrange sur un tel espace énonce quelque chose
que la physique n'utilise pas, et parfois quelque chose de **faux** : la forme
fermée de l'élément de portique exact vient de fonctions cubiques et
quadratiques, pas linéaires.

`MODEL_EMBEDDED` le dit. C'est la troisième combinaison des deux bases :

| espace | géométrie | champ |
|---|---|---|
| `LAGRANGE1` / `LAGRANGE2` | Lagrange | la même |
| `HERMITE3` | Lagrange-1 | Hermite, 4 fonctions |
| `MODEL_EMBEDDED` | Lagrange | **absente, par déclaration** |

La géométrie reste entièrement définie — coordonnées, jacobien, mesure — donc
toute la plomberie d'assemblage est inchangée : champ matériau, sortie de
comportement, layout, coloriage, dispersion parallèle. Seule l'interpolation de
**l'inconnue** n'appartient pas à l'espace.

#### Le refus a des dents

Un accesseur de champ sur un tel espace **erronne**, en nommant la situation,
plutôt que de rendre la base géométrique. La distinction compte *parce que* la
base géométrique est disponible et paraîtrait plausible : c'est exactement le
repli silencieux que cette variante existe pour empêcher.

Ce qui suppose que les consommateurs posent la question au bon endroit. Les
opérateurs qui interpolent une **inconnue** sont donc passés à l'accesseur de
champ :

| opérateur | ce qu'il interpole |
|---|---|
| `node_field::flux` | la fonction test d'une charge répartie |
| `element_field::deformation` | `u_r`, pour la déformation orthoradiale |
| `element_field::interp_to_gauss` | nodal → points de Gauss |
| `measure::integral` | le champ intégré |

Sur une poutre, le premier est le plus parlant : la charge répartie cohérente
vaut `qL/2` **et** `qL²/12` en moment, ce qui suppose de connaître la base. Un
repli linéaire donnerait `qL/2` sans moment — plausible, et faux. Refuser est la
bonne réponse.

**La visualisation, elle, reste sur la base géométrique**, et délibérément : ce
qu'on colorie est ce qu'on dessine — un segment droit, une facette plane — donc
la couleur doit varier le long du tracé. C'est une image, pas une valeur
calculée.

#### Et ce que la formulation doit alors fournir elle-même

La reconstruction des efforts passe à la charge de la formulation, puisque
l'espace ne peut plus interpoler. `beam_deformation` évalue donc les
déformations depuis les fonctions de forme **de l'élément**, ce qui l'oblige à
recevoir le matériau : `Φ` en dépend. C'est le prix, et il est juste — on ne
peut pas reconstituer la courbure d'une poutre sans connaître sa raideur de
cisaillement.

### Dérivées de référence

Les **dérivées de référence** \\( \partial N_i / \partial \xi_k \\) suivent par dérivation directe. Pour Lagrange-1 sur les simplexes (TRI3, TET4) et sur SEG2, ces dérivées sont **constantes** sur l'élément. Sur QUA4 et HEX8 — et sur tous les types quadratiques — elles sont polynomiales en les coordonnées de référence.

En dérivant la partition de l'unité, on obtient également \\( \sum_i \partial N_i / \partial \xi_k = 0 \\) pour chaque direction de référence \\( k \\). Cette identité est aussi testée.

### Stockage flat

Le buffer plat des dérivées de référence retourné par `Interpolation::dshape_dxi(et, &xi)` est de longueur \\( n_\text{nodes} \times d_r \\) (avec \\( d_r = \dim \hat{K} \\)) et row-major :
\\[
\mathtt{dN}[i \times d_r + k] = \frac{\partial N_i}{\partial \xi_k}(\xi)
\\]

## Théorie : quadrature de Gauss

Pour intégrer une fonction \\( f \\) sur l'élément physique, on remonte à l'élément de référence par le changement de variables \\( \mathbf{x} = \chi(\xi) \\) :

\\[
\int_K f(\mathbf{x})\\, d\mathbf{x}
= \int_{\hat{K}} f(\chi(\xi))\\, |J(\xi)|\\, d\xi
\approx \sum_{g=1}^{n_g} w_g\\, f(\chi(\xi_g))\\, |J(\xi_g)|
\\]

avec \\( |J| \\) le **déterminant** (au sens généralisé, défini ci-dessous) du Jacobien. Le couple \\( (\xi_g, w_g) \\) est la règle de quadrature.

pyrucast utilise une règle « par défaut » par type d'élément, calibrée pour **intégrer exactement la matrice de masse Lagrange-1** sur un élément à géométrie droite. Les règles sont :

| ElementType | \\( n_g \\) | Règle | Exactitude polynomiale |
|---|---:|---|---|
| `SEG2` | 2 | Gauss-Legendre sur \\([-1, +1]\\) : \\( \xi_g = \pm 1/\sqrt{3} \\), \\( w_g = 1 \\) | \\( \deg \le 3 \\) |
| `TRI3` | 3 | Hammer mid-edge sur \\( \hat{K} \\) : \\( (\tfrac{1}{2}, 0), (\tfrac{1}{2}, \tfrac{1}{2}), (0, \tfrac{1}{2}) \\), \\( w_g = 1/6 \\) | \\( \deg \le 2 \\) |
| `QUA4` | 4 | Produit tensoriel 2×2 de Gauss-Legendre : \\( \xi_g = (\pm 1/\sqrt{3}, \pm 1/\sqrt{3}) \\), \\( w_g = 1 \\) | \\( \deg \le 3 \\) par direction |
| `TET4` | 4 | Hammer : \\( \alpha = \tfrac{5 - \sqrt{5}}{20} \\), \\( \beta = \tfrac{5 + 3\sqrt{5}}{20} \\), points permutations, \\( w_g = 1/24 \\) | \\( \deg \le 2 \\) |
| `PYRA5` | 8 | Produit **conique** : 2×2 Gauss-Legendre sur la section carrée × Gauss-Jacobi 2 points en \\( \zeta \\) (poids \\( (1-\zeta)^2 \\), nœuds \\( \tfrac13 \mp \tfrac{\sqrt{10}}{15} \\)) | \\( \deg \le 2 \\) |
| `PENTA6` | 6 | Produit tensoriel de la règle TRI3 (3 points, \\( w = 1/6 \\)) et de Gauss-Legendre 2 points sur \\( \zeta \in [0, 1] \\) (\\( \zeta_g = \tfrac{1}{2} \pm \tfrac{1}{2\sqrt{3}} \\), \\( w = 1/2 \\)) | \\( \deg \le 2 \\) en \\( (\xi, \eta) \\), \\( \le 3 \\) en \\( \zeta \\) |
| `HEX8` | 8 | Produit tensoriel 2×2×2 de Gauss-Legendre : \\( \xi_g = (\pm 1/\sqrt{3})^3 \\), \\( w_g = 1 \\) | \\( \deg \le 3 \\) par direction |
| `SEG3` | 3 | Gauss-Legendre 3 points | \\( \deg \le 5 \\) |
| `TRI6` | 6 | Règle symétrique degré 4 (Dunavant) | \\( \deg \le 4 \\) |
| `QUA8` | 9 | Produit tensoriel 3×3 | \\( \deg \le 5 \\) par direction |
| `QUA9` | 9 | Produit tensoriel 3×3 | \\( \deg \le 5 \\) par direction |
| `TET10` | 11 | Règle de Keast degré 4 (un poids négatif) | \\( \deg \le 4 \\) |
| `PENTA15` | 18 | Produit tensoriel TRI6 × Gauss 3 points sur \\( \zeta \\) | \\( \deg \le 4 \\) / \\( \le 5 \\) en \\( \zeta \\) |
| `HEX20` | 27 | Produit tensoriel 3×3×3 | \\( \deg \le 5 \\) par direction |
| `HEX27` | 27 | Produit tensoriel 3×3×3 | \\( \deg \le 5 \\) par direction |

Les types quadratiques utilisent une règle exacte pour leur matrice de masse (degré 4) sur géométrie droite ; l'exactitude des règles custom TRI6 et TET10 est vérifiée par des tests d'intégration de monômes.

La somme des poids vaut le volume de l'élément de référence : 2 pour SEG2/SEG3, 1/2 pour TRI3/TRI6, 4 pour QUA4/QUA8/QUA9, 1/6 pour TET4/TET10, 4/3 pour PYRA5, 1/2 pour PENTA6/PENTA15, 8 pour HEX8/HEX20/HEX27 (vérifié par les tests, pour tous les types à la fois).

## Théorie : Jacobien et grandeurs physiques

### Jacobien

Le **Jacobien** de la transformation \\( \chi \\) est la matrice de dérivées
\\[
J_{a,k}(\xi) = \frac{\partial x_a}{\partial \xi_k}
= \sum_{i=1}^{n} \mathbf{x}_{i,a}\\, \frac{\partial N_i}{\partial \xi_k}(\xi)
\quad
\text{de taille } d_s \times d_r
\\]
où \\( a \in \{0, \dots, d_s - 1\} \\) parcourt les directions physiques et \\( k \in \{0, \dots, d_r - 1\} \\) les directions de référence. Le buffer plat retourné par `SubFiniteElementSpace::jacobian(cell, g)` suit la convention row-major \\( \mathtt{J}[a \times d_r + k] = J_{a,k} \\).

### Cas standard : \\( d_s = d_r \\)

Quand le maillage et son espace ambiant ont la même dimension (par exemple TRI3 dans une `Coords` 2D), \\( J \\) est carrée. Le déterminant ordinaire \\( \det(J) \\) mesure la dilatation locale du volume ; son **valeur absolue** intervient dans l'intégration. La fonction `SubFiniteElementSpace::det_jacobian` retourne \\( |\det(J)| \\).

La dérivation des fonctions de forme par rapport aux coordonnées physiques utilise l'inverse de \\( J \\) :
\\[
\frac{\partial N_i}{\partial x_a}
= \sum_{k=1}^{d_r} (J^{-1})_{k, a} \\, \frac{\partial N_i}{\partial \xi_k}
\quad \Longleftrightarrow \quad
\nabla_x N_i = J^{-T}\\, \nabla_\xi N_i
\\]

### Cas manifold : \\( d_s > d_r \\)

Un sous-maillage peut être **plongé** dans un espace de dimension supérieure : SEG2 dans une `Coords` 2D ou 3D (contour, courbe), TRI3 dans une `Coords` 3D (surface plongée). C'est exactement ce que produit `triangulate_surface` quand on lui donne un contour 3D plan.

Dans ce cas, \\( J \\) est rectangulaire (taille \\( d_s \times d_r \\)). Le déterminant standard n'a plus de sens, mais on peut définir la **métrique tirée en arrière** :
\\[
G(\xi) = J(\xi)^T\\, J(\xi) \quad \text{de taille } d_r \times d_r
\\]
\\( G \\) est symétrique définie positive (sous condition de non-dégénérescence). L'élément de mesure devient
\\[
d\mu = \sqrt{\det G}\\, d\xi
\\]
qui s'utilise comme \\( |J| \\) dans la quadrature. La fonction `det_jacobian` retourne ce \\( \sqrt{\det G} \\) — toujours positif par construction.

Le **gradient tangent** d'un champ sur la surface (la projection du vrai gradient sur l'espace tangent) est donné par la pseudo-inverse :
\\[
\nabla_s N_i = J\\, G^{-1}\\, \nabla_\xi N_i
\\]
La fonction `SubFiniteElementSpace::dn_dx` retourne ces composantes, dans le repère ambiant à \\( d_s \\) dimensions. Pour \\( d_s = d_r \\), ces formules se réduisent à celles du cas standard (\\( J G^{-1} = J^{-T} \\)).

### Combinaisons valides

Le couple \\( (d_r, d_s) \\) possible pour notre v0 :

| ElementType | \\( d_r \\) | \\( d_s \\) valides |
|---|---:|---|
| SEG2 | 1 | 1, 2, 3 |
| TRI3 | 2 | 2, 3 |
| QUA4 | 2 | 2, 3 |
| TET4 | 3 | 3 |
| PYRA5 | 3 | 3 |
| PENTA6 | 3 | 3 |
| HEX8 | 3 | 3 |
| SEG3 | 1 | 1, 2, 3 |
| TRI6 | 2 | 2, 3 |
| QUA8 | 2 | 2, 3 |
| QUA9 | 2 | 2, 3 |
| TET10 | 3 | 3 |
| PENTA15 | 3 | 3 |
| HEX20 | 3 | 3 |
| HEX27 | 3 | 3 |

\\( d_s < d_r \\) n'a pas de sens (impossible de définir un élément 2D dans un espace 1D) ; le constructeur de `SubFiniteElementSpace` rejette ce cas.

## Stratégie de stockage : invariant vs variant à la déformation

Le maillage support d'un `FiniteElementSpace` est topologiquement figé, mais ses **coordonnées** peuvent évoluer (déplacement de maillage, mise à jour incrémentale). pyrucast scinde donc les grandeurs en deux catégories selon leur invariance vis-à-vis de cette déformation :

| Grandeur | Variant avec les coordonnées ? | Stratégie |
|---|---|---|
| Points de Gauss \\( \xi_g \\), poids \\( w_g \\) | non (référence pure) | **précalculés** dans le `SubFiniteElementSpace` |
| \\( N_i(\xi_g) \\), \\( \partial N_i / \partial \xi_k(\xi_g) \\) | non (référence pure) | **précalculés** dans le `SubFiniteElementSpace` |
| Jacobien \\( J(\xi_g) \\) sur chaque cellule | oui | **calculé à la volée** |
| \\( |J|(\xi_g) \\), \\( \partial N_i / \partial x_a(\xi_g) \\) | oui | **calculé à la volée** |

Ce choix donne deux propriétés importantes :

1. **Empreinte mémoire indépendante du nombre de cellules.** Un `SubFiniteElementSpace` ne stocke que de l'ordre de \\( n_g \times n_\text{nodes} \times d_r \\) flottants — quelques centaines au plus par sous-espace. À comparer aux GB qu'un précalcul des Jacobiens demanderait sur un maillage 3D fin.
2. **Robustesse au déplacement.** Réécrire les coordonnées dans la `Coords` (par exemple via `Coords::set_position`) suffit à mettre à jour automatiquement toutes les évaluations de \\( J \\), \\( |J| \\) et \\( \partial N_i / \partial x_a \\) — pas d'invalidation à signaler, pas de cache à reconstruire.

Le coût est CPU plutôt que mémoire : chaque appel à `jacobian(cell, g)` recalcule la somme \\( J = \sum_i \mathbf{x}_i\\, \nabla_\xi N_i \\). En pratique, l'assemblage matrice-élémentaire procède **cellule par cellule** : on calcule \\( J \\), \\( |J| \\), \\( \nabla_x N_i \\) une fois par couple (cellule, Gauss), puis on les réutilise pour tous les termes intégrés. Le surcoût reste donc proportionnel à \\( n_\text{cells} \times n_g \\) — soit le minimum incompressible — et non à \\( n_\text{cells} \times n_g \times n_\text{termes} \\).

Si une mesure montrait un jour que ce recalcul devient un goulot d'étranglement, un cache invalidé sur incrément d'un compteur de version du `Coords` pourrait être ajouté sans changer l'API publique — déclenché par la mesure, pas par anticipation.

## Validation à la construction

`SubFiniteElementSpace::new` rejette à la création :

- un sous-maillage de type `POI1` (pas de repère de référence) ;
- un couple `(ElementType, Interpolation)` non supporté (par exemple `TRI3` + `Lagrange2` tant que `Lagrange2` n'existe pas) ;
- un couple `(ElementType, QuadratureRule)` non supporté ;
- \\( d_s < d_r \\) (incompatible avec la définition du Jacobien).

`FiniteElementSpace::with` (et donc `lagrange1`, `new`) vérifie en plus :

- que le maillage contient au moins un sous-maillage ;
- que la longueur de la liste `(interpolation, quadrature)` correspond au nombre de sous-maillages ;
- que chaque `SubFiniteElementSpace` se construit sans erreur.

Le déterminant du Jacobien n'est **pas** vérifié à la construction : un élément dégénéré ou inversé ne sera détecté qu'à la première évaluation de `det_jacobian`. Cette validation paresseuse permet de construire un FE space sans toucher aux coordonnées et reste correcte vis-à-vis du déplacement de maillage ultérieur (un élément valide peut devenir dégénéré après un déplacement et inversement).

## API Rust

Constructeur principal — Lagrange-1 partout, quadrature de Gauss par défaut :

```rust,ignore
use pyrucast::coords::Coords;
use pyrucast::atoms::ElementType;
use pyrucast::containers::finite_element_space::FiniteElementSpace;
use pyrucast::containers::mesh::{Mesh, SubMesh};
use pyrucast::atoms::Node;
use pyrucast::store::{insert, read};

let coords = insert(Coords::new(2).unwrap());
let a = Node::create_in(coords.clone(), &[0.0, 0.0]).unwrap();
let b = Node::create_in(coords.clone(), &[2.0, 0.0]).unwrap();
let c = Node::create_in(coords.clone(), &[0.0, 2.0]).unwrap();

let mut mesh = Mesh::from_submesh(SubMesh::new(coords, ElementType::TRI3));
mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
let sub = fes.get(0).unwrap();
let s = read(&sub).unwrap();
assert_eq!(s.gauss_count(), 3);
// Le triangle (0,0), (2,0), (0,2) a |J| = 4 partout :
// mapping affine, det(J) = 4 = 2 × aire physique du triangle (1/2 × 2 × 2).
for g in 0..s.gauss_count() {
    let dj = s.det_jacobian(0, g).unwrap();
    assert!((dj - 4.0).abs() < 1e-12);
}
```

Constructeur explicite — utile pour mélanger les interpolations / quadratures par sous-maillage :

```rust,ignore
use pyrucast::atoms::Interpolation;
use pyrucast::atoms::QuadratureRule;

let fes = FiniteElementSpace::with(
    mesh_h,
    &[
        (Interpolation::Lagrange1, QuadratureRule::Gauss),
        (Interpolation::Lagrange1, QuadratureRule::Gauss),
    ],
).unwrap();
```

Évaluation des grandeurs sur une cellule :

```rust,ignore
let s = read(&sub).unwrap();
for cell_idx in 0..s.cell_count().unwrap() {
    for g in 0..s.gauss_count() {
        let n  = s.n_at_g(g).unwrap();          // N_i(ξ_g)
        let dn = s.dn_at_g(g).unwrap();         // ∂N_i/∂ξ_k(ξ_g)
        let jac     = s.jacobian(cell_idx, g).unwrap();
        let det_j   = s.det_jacobian(cell_idx, g).unwrap();
        let dn_dx   = s.dn_dx(cell_idx, g).unwrap();
        // … utiliser ces buffers dans l'assemblage matrice-élémentaire …
    }
}
```

## Déplacement de maillage : exemple

Après modification des coordonnées dans la `Coords`, les évaluations à la volée reflètent automatiquement le nouvel état :

```rust,ignore
use pyrucast::store::{read, write};

// SEG2 initial : nœuds en x=0 et x=1 → |J| = 0.5 (longueur 1 sur [-1,+1]).
let dj_before = read(&sub).unwrap().det_jacobian(0, 0).unwrap();
assert!((dj_before - 0.5).abs() < 1e-12);

// Étirement : on déplace le second nœud en x=4 → |J| = 2.0 (longueur 4 sur [-1,+1]).
write(&coords).unwrap().set_position(b.id(), &[4.0, 0.0]).unwrap();
let dj_after = read(&sub).unwrap().det_jacobian(0, 0).unwrap();
assert!((dj_after - 2.0).abs() < 1e-12);
```

## API Python

L'objet est exposé sous le nom `pyrucast.FiniteElementSpace`, avec les
sous-espaces accessibles via `pyrucast.SubFiniteElementSpace`. Les interpolations et
règles de quadrature sont passées en chaînes de caractères
(`"LAGRANGE1"`, `"GAUSS"`).

```python
import pyrucast

c = pyrucast.Coords(dim=2)
n0 = c.add_node([0.0, 0.0])
n1 = c.add_node([2.0, 0.0])
n2 = c.add_node([0.0, 2.0])

mesh = pyrucast.Mesh(c, "TRI3")
mesh.unit().add_cell([n0, n1, n2])

# Constructeur par défaut : Lagrange1 + Gauss partout.
fes = pyrucast.FiniteElementSpace(mesh)
assert len(fes) == 1  # 1 sous-espace = 1 sous-maillage
sub = fes[0]  # vue typée du sous-espace 0
assert sub.element_type == "TRI3"
assert sub.interpolation == "LAGRANGE1"
assert sub.quadrature == "GAUSS"
assert sub.gauss_count() == 3
assert sub.space_dim == 2
assert sub.ref_dim == 2

# Évaluations à un point de Gauss donné.
for g in range(sub.gauss_count()):
    print(sub.gauss_xi(g), sub.gauss_weight(g))
    print(sub.n_at_g(g))  # N_i(ξ_g), flat
    print(sub.dn_at_g(g))  # ∂N_i/∂ξ_j(ξ_g), flat

# Grandeurs physiques (à la volée) sur la cellule 0.
print(sub.jacobian(0, 0))  # J, flat row-major
print(sub.det_jacobian(0, 0))  # |J|, scalaire
print(sub.dn_dx(0, 0))  # ∂N_i/∂x_a, flat row-major
```

Variantes de construction :

```python
# Même Lagrange1 + même Gauss pour tous les sous-maillages, explicite.
fes = pyrucast.FiniteElementSpace(mesh, interpolation="LAGRANGE1", quadrature="GAUSS")

# Forme « class method » équivalente au constructeur par défaut.
fes = pyrucast.FiniteElementSpace.lagrange1(mesh)

# (Interpolation, quadrature) explicites par sous-maillage.
fes = pyrucast.FiniteElementSpace.with_choices(mesh, [("LAGRANGE1", "GAUSS")])
```

Déplacement du maillage : le Jacobien reflète automatiquement les
coordonnées courantes du `Coords` — pas de cache à invalider.

```python
print(sub.det_jacobian(0, 0))  # |J| initial

# Déplacement d'un nœud → toutes les évaluations à venir voient les
# nouvelles coordonnées.
n1.set_position([4.0, 0.0])
print(sub.det_jacobian(0, 0))  # |J| recalculé
```

## Limitations actuelles

- Quatre interpolations : `Lagrange1` (types linéaires), `Lagrange2` (types quadratiques `SEG3`, `TRI6`, `QUA8`, `QUA9`, `TET10`, `PENTA15`, `HEX20`, `HEX27`), `Hermite3` (C¹, sur `SEG2` seul) et `ModelEmbedded` (aucune base de champ : la formulation possède la sienne). Pour les familles de Lagrange le **degré** doit correspondre au type d'élément : un maillage quadratique se pose avec `interpolation="LAGRANGE2"` (le constructeur par défaut `LAGRANGE1` refuse un type quadratique, et inversement). `HERMITE3`, lui, se pose sur un `SEG2` **sans** en être le degré — c'est le cas sous-paramétrique. Les ordres supérieurs (Lagrange-3…) restent à venir.
- Deux quadratures : `QuadratureRule::Gauss` (la règle standard par défaut par `ElementType`) et `QuadratureRule::Reduced` (**intégration réduite** : un point au centroïde, exact pour les constantes — utilisée par exemple pour le terme de cisaillement de la [poutre de Timoshenko](mecanique/timoshenko.md), anti-verrouillage). Les variantes d'ordre supérieur viendront comme nouvelles variantes. Voir le [tableau croisé quadrature × élément](elements/index.md#catalogue-de-quadrature) pour la compatibilité couple par couple.
- `POI1` n'est pas un élément fini (pas de repère de référence). Un sous-maillage POI1 dans le maillage support fait échouer la construction du `FiniteElementSpace`.
- Pas encore de cache invalidable des grandeurs physiques (\\( J \\), \\( |J| \\), \\( \nabla_x N_i \\)) : tout est recalculé à la volée. Une optimisation à base d'invalidation par compteur de version pourra être ajoutée si la mesure le justifie, sans changement d'API.
