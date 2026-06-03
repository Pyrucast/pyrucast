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

## Conventions de l'élément de référence

Chaque `ElementType` fixe son repère de référence \\( \xi \\) et la numérotation locale de ses nœuds. Ces conventions sont aussi documentées dans le rustdoc de [`ElementType`](https://docs.rs/) et reproduites ici pour référence centrale.

| ElementType | Repère \\( \xi \\) | Numérotation locale (ordre des nœuds) |
|---|---|---|
| `SEG2` | \\( \xi \in [-1, +1] \\) | nœud 0 en \\( \xi = -1 \\), nœud 1 en \\( \xi = +1 \\) |
| `TRI3` | \\( \xi, \eta \in [0, 1] \\), \\( \xi + \eta \le 1 \\) | \\( (0,0), (1,0), (0,1) \\) — CCW |
| `QUA4` | \\( \xi, \eta \in [-1, +1] \\) | \\( (-1,-1), (1,-1), (1,1), (-1,1) \\) — CCW |
| `TET4` | \\( \xi, \eta, \zeta \in [0, 1] \\), \\( \xi + \eta + \zeta \le 1 \\) | \\( (0,0,0), (1,0,0), (0,1,0), (0,0,1) \\) — face 0-1-2 CCW vue depuis nœud 3 |
| `HEX8` | \\( \xi, \eta, \zeta \in [-1, +1] \\) | face inférieure CCW (nœuds 0..3) puis face supérieure CCW (nœuds 4..7) |

Ces conventions sont cohérentes avec celles déjà imposées ailleurs dans le code : orientation CCW des triangles produits par `Mesh::fill_surface`, ordre HEX8 utilisé par `Mesh::extrude`.

## Théorie : élément isoparamétrique

### Transformation géométrique de référence

Soit un élément avec \\( n \\) nœuds, de coordonnées physiques \\( \mathbf{x}_1, \dots, \mathbf{x}_n \in \mathbb{R}^{d_s} \\) (avec \\( d_s \\) la dimension géométrique de la `Configuration`). La **transformation géométrique** \\( \chi : \hat{K} \to K \\) qui envoie l'élément de référence \\( \hat{K} \\) sur l'élément physique \\( K \\) est interpolée par les mêmes fonctions de forme :

\\[
\mathbf{x}(\xi) = \chi(\xi) = \sum_{i=1}^{n} N_i(\xi)\, \mathbf{x}_i
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
N_i(\xi, \eta, \zeta) = \tfrac{1}{8}\,
(1 + \xi_i\, \xi)\,
(1 + \eta_i\, \eta)\,
(1 + \zeta_i\, \zeta)
\\]

Une propriété immédiate, vérifiée par les tests unitaires : \\( \sum_i N_i(\xi) = 1 \\) en tout \\( \xi \\) (partition de l'unité).

### Dérivées de référence

Les **dérivées de référence** \\( \partial N_i / \partial \xi_k \\) suivent par dérivation directe. Pour Lagrange-1 sur les simplexes (TRI3, TET4) et sur SEG2, ces dérivées sont **constantes** sur l'élément. Sur QUA4 et HEX8, elles sont polynomiales en les coordonnées de référence.

En dérivant la partition de l'unité, on obtient également \\( \sum_i \partial N_i / \partial \xi_k = 0 \\) pour chaque direction de référence \\( k \\). Cette identité est aussi testée.

### Stockage flat

Le buffer plat des dérivées de référence retourné par `Interpolation::dshape_dxi(et, &xi)` est de longueur \\( n_\text{nodes} \times d_r \\) (avec \\( d_r = \dim \hat{K} \\)) et row-major :
\\[
\mathtt{dN}[i \times d_r + k] = \frac{\partial N_i}{\partial \xi_k}(\xi)
\\]

## Théorie : quadrature de Gauss

Pour intégrer une fonction \\( f \\) sur l'élément physique, on remonte à l'élément de référence par le changement de variables \\( \mathbf{x} = \chi(\xi) \\) :

\\[
\int_K f(\mathbf{x})\, d\mathbf{x}
= \int_{\hat{K}} f(\chi(\xi))\, |J(\xi)|\, d\xi
\approx \sum_{g=1}^{n_g} w_g\, f(\chi(\xi_g))\, |J(\xi_g)|
\\]

avec \\( |J| \\) le **déterminant** (au sens généralisé, défini ci-dessous) du Jacobien. Le couple \\( (\xi_g, w_g) \\) est la règle de quadrature.

pyrucast utilise une règle « par défaut » par type d'élément, calibrée pour **intégrer exactement la matrice de masse Lagrange-1** sur un élément à géométrie droite. Les règles sont :

| ElementType | \\( n_g \\) | Règle | Exactitude polynomiale |
|---|---:|---|---|
| `SEG2` | 2 | Gauss-Legendre sur \\([-1, +1]\\) : \\( \xi_g = \pm 1/\sqrt{3} \\), \\( w_g = 1 \\) | \\( \deg \le 3 \\) |
| `TRI3` | 3 | Hammer mid-edge sur \\( \hat{K} \\) : \\( (\tfrac{1}{2}, 0), (\tfrac{1}{2}, \tfrac{1}{2}), (0, \tfrac{1}{2}) \\), \\( w_g = 1/6 \\) | \\( \deg \le 2 \\) |
| `QUA4` | 4 | Produit tensoriel 2×2 de Gauss-Legendre : \\( \xi_g = (\pm 1/\sqrt{3}, \pm 1/\sqrt{3}) \\), \\( w_g = 1 \\) | \\( \deg \le 3 \\) par direction |
| `TET4` | 4 | Hammer : \\( \alpha = \tfrac{5 - \sqrt{5}}{20} \\), \\( \beta = \tfrac{5 + 3\sqrt{5}}{20} \\), points permutations, \\( w_g = 1/24 \\) | \\( \deg \le 2 \\) |
| `HEX8` | 8 | Produit tensoriel 2×2×2 de Gauss-Legendre : \\( \xi_g = (\pm 1/\sqrt{3})^3 \\), \\( w_g = 1 \\) | \\( \deg \le 3 \\) par direction |

La somme des poids vaut le volume de l'élément de référence : 2 pour SEG2, 1/2 pour TRI3, 4 pour QUA4, 1/6 pour TET4, 8 pour HEX8 (vérifié par les tests).

## Théorie : Jacobien et grandeurs physiques

### Jacobien

Le **Jacobien** de la transformation \\( \chi \\) est la matrice de dérivées
\\[
J_{a,k}(\xi) = \frac{\partial x_a}{\partial \xi_k}
= \sum_{i=1}^{n} \mathbf{x}_{i,a}\, \frac{\partial N_i}{\partial \xi_k}(\xi)
\quad
\text{de taille } d_s \times d_r
\\]
où \\( a \in \{0, \dots, d_s - 1\} \\) parcourt les directions physiques et \\( k \in \{0, \dots, d_r - 1\} \\) les directions de référence. Le buffer plat retourné par `SubFiniteElementSpace::jacobian(cell, g)` suit la convention row-major \\( \mathtt{J}[a \times d_r + k] = J_{a,k} \\).

### Cas standard : \\( d_s = d_r \\)

Quand le maillage et son espace ambiant ont la même dimension (par exemple TRI3 dans une `Configuration` 2D), \\( J \\) est carrée. Le déterminant ordinaire \\( \det(J) \\) mesure la dilatation locale du volume ; son **valeur absolue** intervient dans l'intégration. La fonction `SubFiniteElementSpace::det_jacobian` retourne \\( |\det(J)| \\).

La dérivation des fonctions de forme par rapport aux coordonnées physiques utilise l'inverse de \\( J \\) :
\\[
\frac{\partial N_i}{\partial x_a}
= \sum_{k=1}^{d_r} (J^{-1})_{k, a} \, \frac{\partial N_i}{\partial \xi_k}
\quad \Longleftrightarrow \quad
\nabla_x N_i = J^{-T}\, \nabla_\xi N_i
\\]

### Cas manifold : \\( d_s > d_r \\)

Un sous-maillage peut être **plongé** dans un espace de dimension supérieure : SEG2 dans une `Configuration` 2D ou 3D (contour, courbe), TRI3 dans une `Configuration` 3D (surface plongée). C'est exactement ce que produit `Mesh::fill_surface` quand on lui donne un contour 3D plan.

Dans ce cas, \\( J \\) est rectangulaire (taille \\( d_s \times d_r \\)). Le déterminant standard n'a plus de sens, mais on peut définir la **métrique tirée en arrière** :
\\[
G(\xi) = J(\xi)^T\, J(\xi) \quad \text{de taille } d_r \times d_r
\\]
\\( G \\) est symétrique définie positive (sous condition de non-dégénérescence). L'élément de mesure devient
\\[
d\mu = \sqrt{\det G}\, d\xi
\\]
qui s'utilise comme \\( |J| \\) dans la quadrature. La fonction `det_jacobian` retourne ce \\( \sqrt{\det G} \\) — toujours positif par construction.

Le **gradient tangent** d'un champ sur la surface (la projection du vrai gradient sur l'espace tangent) est donné par la pseudo-inverse :
\\[
\nabla_s N_i = J\, G^{-1}\, \nabla_\xi N_i
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
| HEX8 | 3 | 3 |

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
2. **Robustesse au déplacement.** Réécrire les coordonnées dans la `Configuration` (par exemple via `Configuration::set_coord`) suffit à mettre à jour automatiquement toutes les évaluations de \\( J \\), \\( |J| \\) et \\( \partial N_i / \partial x_a \\) — pas d'invalidation à signaler, pas de cache à reconstruire.

Le coût est CPU plutôt que mémoire : chaque appel à `jacobian(cell, g)` recalcule la somme \\( J = \sum_i \mathbf{x}_i\, \nabla_\xi N_i \\). En pratique, l'assemblage matrice-élémentaire procède **cellule par cellule** : on calcule \\( J \\), \\( |J| \\), \\( \nabla_x N_i \\) une fois par couple (cellule, Gauss), puis on les réutilise pour tous les termes intégrés. Le surcoût reste donc proportionnel à \\( n_\text{cells} \times n_g \\) — soit le minimum incompressible — et non à \\( n_\text{cells} \times n_g \times n_\text{termes} \\).

Si une mesure montrait un jour que ce recalcul devient un goulot d'étranglement, un cache invalidé sur incrément d'un compteur de version du `Configuration` pourrait être ajouté sans changer l'API publique. Phase 6, déclenché par la mesure.

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
use pyrucast::mesh::configuration::Configuration;
use pyrucast::mesh::element_type::ElementType;
use pyrucast::finite_element_space::FiniteElementSpace;
use pyrucast::mesh::{Mesh, SubMesh};
use pyrucast::mesh::node::Node;
use pyrucast::store::{insert, with};

let cfg = insert(Configuration::new(2).unwrap());
let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
let b = Node::create_in(cfg.clone(), &[2.0, 0.0]).unwrap();
let c = Node::create_in(cfg.clone(), &[0.0, 2.0]).unwrap();

let mut mesh = Mesh::from_submesh(SubMesh::new(cfg, ElementType::TRI3));
mesh.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

let fes = FiniteElementSpace::lagrange1(&mesh).unwrap();
let sub = fes.subspace(0).unwrap();
with(&sub, |s| {
    assert_eq!(s.gauss_count(), 3);
    // Le triangle (0,0), (2,0), (0,2) a |J| = 4 partout :
    // mapping affine, det(J) = 4 = 2 × aire physique du triangle (1/2 × 2 × 2).
    for g in 0..s.gauss_count() {
        let dj = s.det_jacobian(0, g).unwrap();
        assert!((dj - 4.0).abs() < 1e-12);
    }
}).unwrap();
```

Constructeur explicite — utile pour mélanger les interpolations / quadratures par sous-maillage :

```rust,ignore
use pyrucast::finite_element_space::interpolation::Interpolation;
use pyrucast::finite_element_space::quadrature::QuadratureRule;

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
with(&sub, |s| {
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
}).unwrap();
```

## Déplacement de maillage : exemple

Après modification des coordonnées dans la `Configuration`, les évaluations à la volée reflètent automatiquement le nouvel état :

```rust,ignore
use pyrucast::store::with_mut;

// SEG2 initial : nœuds en x=0 et x=1 → |J| = 0.5 (longueur 1 sur [-1,+1]).
let dj_before = with(&sub, |s| s.det_jacobian(0, 0)).unwrap().unwrap();
assert!((dj_before - 0.5).abs() < 1e-12);

// Étirement : on déplace le second nœud en x=4 → |J| = 2.0 (longueur 4 sur [-1,+1]).
with_mut(&cfg, |c| c.set_coord(b.id(), &[4.0, 0.0])).unwrap().unwrap();
let dj_after = with(&sub, |s| s.det_jacobian(0, 0)).unwrap().unwrap();
assert!((dj_after - 2.0).abs() < 1e-12);
```

## API Python

L'objet est exposé sous le nom `pyrucast.FiniteElementSpace`, avec les
sous-espaces accessibles via `pyrucast.SubFiniteElementSpace`. Les interpolations et
règles de quadrature sont passées en chaînes de caractères
(`"LAGRANGE1"`, `"GAUSS"`).

```python
import pyrucast

c = pyrucast.Configuration(dim=2)
n0 = c.add_node([0.0, 0.0])
n1 = c.add_node([2.0, 0.0])
n2 = c.add_node([0.0, 2.0])

mesh = pyrucast.Mesh(c, "TRI3")
mesh.unit().add_cell([n0, n1, n2])

# Constructeur par défaut : Lagrange1 + Gauss partout.
fes = pyrucast.FiniteElementSpace(mesh)
assert len(fes) == 1                          # 1 sous-espace = 1 sous-maillage
sub = fes[0]                                  # ou fes.subspace(0)
assert sub.element_type == "TRI3"
assert sub.interpolation == "LAGRANGE1"
assert sub.quadrature == "GAUSS"
assert sub.gauss_count() == 3
assert sub.space_dim == 2
assert sub.ref_dim == 2

# Évaluations à un point de Gauss donné.
for g in range(sub.gauss_count()):
    print(sub.gauss_xi(g), sub.gauss_weight(g))
    print(sub.n_at_g(g))                      # N_i(ξ_g), flat
    print(sub.dn_at_g(g))                     # ∂N_i/∂ξ_j(ξ_g), flat

# Grandeurs physiques (à la volée) sur la cellule 0.
print(sub.jacobian(0, 0))                     # J, flat row-major
print(sub.det_jacobian(0, 0))                 # |J|, scalaire
print(sub.dn_dx(0, 0))                        # ∂N_i/∂x_a, flat row-major
```

Variantes de construction :

```python
# Même Lagrange1 + même Gauss pour tous les sous-maillages, explicite.
fes = pyrucast.FiniteElementSpace(mesh, interpolation="LAGRANGE1", quadrature="GAUSS")

# Forme « class method » équivalente au constructeur par défaut.
fes = pyrucast.FiniteElementSpace.lagrange1(mesh)

# (Interpolation, quadrature) explicites par sous-maillage.
fes = pyrucast.FiniteElementSpace.with_choices(
    mesh, [("LAGRANGE1", "GAUSS")]
)
```

Déplacement du maillage : le Jacobien reflète automatiquement les
coordonnées courantes du `Configuration` — pas de cache à invalider.

```python
print(sub.det_jacobian(0, 0))                 # |J| initial

# Déplacement d'un nœud → toutes les évaluations à venir voient les
# nouvelles coordonnées.
n1.set_coord([4.0, 0.0])
print(sub.det_jacobian(0, 0))                 # |J| recalculé
```

## Limitations actuelles

- Une seule interpolation : `Lagrange1`. Les éléments quadratiques (TRI6, QUA8, etc.) viendront en parallèle d'une variante `Lagrange2`, conditionnée à l'ajout des `ElementType` correspondants.
- Une seule quadrature : `QuadratureRule::Gauss` (la règle standard par défaut par `ElementType`). Les variantes « intégration réduite » ou « ordre supérieur » seront ajoutées comme nouvelles variantes.
- `POI1` n'est pas un élément fini (pas de repère de référence). Un sous-maillage POI1 dans le maillage support fait échouer la construction du `FiniteElementSpace`.
- Pas encore de cache invalidable des grandeurs physiques (\\( J \\), \\( |J| \\), \\( \nabla_x N_i \\)) : tout est recalculé à la volée. Une optimisation à base d'invalidation par compteur de version pourra être ajoutée si la mesure le justifie, sans changement d'API.
