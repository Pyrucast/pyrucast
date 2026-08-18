# Triangulation : briques mathématiques

Ce chapitre rassemble les fondements mathématiques utilisés par `triangulate_surface` et le module `pyrucast::ops::mesh::triangulation`. Toutes les formules sont écrites avec la convention de pyrucast : points 2D notés \\( P = (x, y) \\), points 3D notés \\( P = (x, y, z) \\), vecteurs en gras, produit scalaire \\( \cdot \\), produit vectoriel \\( \times \\).

## Aire signée d'un polygone 2D (formule du lacet)

Soit \\( P_0, P_1, \dots, P_{n-1} \\) les sommets d'un polygone simple fermé (l'arête \\( P_{n-1} P_0 \\) ferme la boucle implicitement). Son **aire signée** est

\\[
A = \frac{1}{2} \sum_{i=0}^{n-1} \left( x_i\\, y_{i+1} - x_{i+1}\\, y_i \right)
\\]

avec la convention d'indices modulo \\(n\\). Le signe de \\(A\\) encode l'orientation :

- \\( A > 0 \\) : polygone parcouru en **sens trigonométrique** (CCW) ;
- \\( A < 0 \\) : sens **horaire** (CW) ;
- \\( A \approx 0 \\) : polygone dégénéré (sommets colinéaires).

Implémentation : `signed_area` dans `src/ops/mesh/triangulation.rs`.

```rust,ignore
{{#include ../../tests/doc_triangulation.rs:aire_signee}}
```

> `signed_area` n'est pas exposée en Python : elle est utilisée en interne par `triangulate_surface`. Pour obtenir l'aire d'un maillage depuis Python, calculez-la à partir des coordonnées des triangles.

## Ear clipping : usage direct

La fonction `ear_clip_2d` est utilisable indépendamment de `triangulate_surface` sur n'importe quel polygone 2D simple :

```rust,ignore
{{#include ../../tests/doc_triangulation.rs:ear_clip}}
```

## Test d'oreille (ear clipping)

Pour un polygone simple CCW à \\(n\\) sommets, un sommet \\(P_i\\) est une **oreille** si :

1. **Convexité locale** en \\(P_i\\) — le triangle \\((P_{i-1}, P_i, P_{i+1})\\) est CCW :

   \\[
   (P_i - P_{i-1}) \times (P_{i+1} - P_i) > 0
   \\]

   (produit en croix 2D \\( \mathbf{a} \times \mathbf{b} = a_x b_y - a_y b_x \\)).

2. **Cavité vide** — aucun autre sommet du polygone n'est dans le triangle fermé \\((P_{i-1}, P_i, P_{i+1})\\).

L'algorithme retire itérativement une oreille (créant un triangle de sortie et un polygone à \\(n-1\\) sommets) jusqu'à ne plus avoir que 3 sommets. Au total **\\(n - 2\\) triangles** sont produits.

L'orientation est détectée d'abord via \\(A\\) ; si \\(A < 0\\), on parcourt les indices à l'envers pour ramener à du CCW.

## Plan moyen et base locale : usage direct

```rust,ignore
{{#include ../../tests/doc_triangulation.rs:repere_local}}
```

> `newell_normal` et `in_plane_basis` ne sont pas exposées en Python. Elles sont utilisées en interne par `triangulate_surface` pour les configurations 3D.

## Plan moyen d'un polygone 3D : méthode de Newell

Soit un polygone à \\(n\\) sommets \\(P_0, \dots, P_{n-1}\\) approximativement coplanaires. La **normale de Newell** est obtenue par somme signée d'arêtes consécutives :

\\[
\vec{n} = \sum_{i=0}^{n-1}
\begin{pmatrix}
(y_i - y_{i+1})(z_i + z_{i+1}) \\\\
(z_i - z_{i+1})(x_i + x_{i+1}) \\\\
(x_i - x_{i+1})(y_i + y_{i+1})
\end{pmatrix}
\\]

**Propriétés clefs** :

- chaque composante de \\(\vec{n}\\) vaut **2 × l'aire signée du polygone projeté sur le plan coordonné correspondant** (lacet généralisé en 3D) ;
- \\(\vec{n}\\) est invariant par translation des sommets ;
- la direction de \\(\vec{n}/\|\vec{n}\|\\) suit la **règle de la main droite** par rapport au sens de parcours.

Si \\(\|\vec{n}\| \approx 0\\), le polygone est dégénéré (colinéaire ou auto-recouvrant) — `newell_normal` renvoie alors `None`.

## Base orthonormée du plan (Gram-Schmidt)

Étant donnée la normale unitaire \\(\hat{n}\\), on construit une base directe \\((\vec{u}, \vec{v}, \hat{n})\\) :

1. On choisit l'axe canonique \\(\vec{e}\\) le moins aligné avec \\(\hat{n}\\) (composante absolue minimale).
2. On orthogonalise par Gram-Schmidt :

   \\[
   \vec{u}' = \vec{e} - (\vec{e} \cdot \hat{n})\\, \hat{n},
   \qquad
   \vec{u} = \frac{\vec{u}'}{\|\vec{u}'\|}
   \\]

3. On complète :

   \\[
   \vec{v} = \hat{n} \times \vec{u}
   \\]

Par construction \\( \|\vec{u}\| = \|\vec{v}\| = 1 \\), \\( \vec{u} \cdot \vec{v} = \vec{u} \cdot \hat{n} = \vec{v} \cdot \hat{n} = 0 \\), et \\( \vec{u} \times \vec{v} = \hat{n} \\) (triedre direct).

## Projection orthogonale et critère de planéité

Un point 3D \\(P\\) est projeté dans le plan local d'origine \\(O\\) (centroïde du contour) par :

\\[
p_u = (P - O) \cdot \vec{u},
\qquad
p_v = (P - O) \cdot \vec{v}
\\]

La **distance algébrique** de \\(P\\) au plan est

\\[
d = (P - O) \cdot \hat{n}
\\]

Le contour est jugé « plan » si

\\[
\max_i |d_i| \le \varepsilon \cdot \mathrm{diag}, \qquad \varepsilon = 10^{-6}
\\]

où `diag` est la longueur de la diagonale de la AABB de l'ensemble des sommets. La tolérance relative \\(10^{-6}\\) tolère le bruit numérique sans laisser passer une vraie courbure 3D.

## Pipeline Delaunay et CDT : usage direct en Rust

Les fonctions du module `pyrucast::ops::mesh::triangulation` sont utilisables indépendamment du système `Mesh`.

### Delaunay pur

```rust,ignore
{{#include ../../tests/doc_triangulation.rs:delaunay}}
```

### Delaunay contraint (CDT) avec trous

```rust,ignore
{{#include ../../tests/doc_triangulation.rs:polygone_troue}}
```

### CDT avec raffinement de Ruppert

```rust,ignore
{{#include ../../tests/doc_triangulation.rs:raffinement}}
```

> Ces fonctions renvoient des indices dans le tableau de points fourni en entrée (plus les Steiner éventuels). Elles ne touchent pas à la `Coords` — `triangulate_surface` se charge de la conversion vers les `NodeId`.

## Triangulation de Delaunay : la propriété du cercle vide

Une triangulation \\(T\\) d'un nuage de points \\(\{P_0, \dots, P_{n-1}\}\\) est dite **de Delaunay** si, pour tout triangle \\((P_a, P_b, P_c) \in T\\), le disque circonscrit ne contient strictement aucun autre point du nuage.

Cette propriété équivaut à **maximiser l'angle minimum** sur l'ensemble des triangulations possibles — c'est pourquoi Delaunay est la triangulation de référence pour le FEM : elle évite naturellement les triangles très allongés.

## Test in-circumcircle

Pour un triangle \\((A, B, C)\\) orienté CCW, le point \\(D\\) est **strictement dans** son disque circonscrit si et seulement si

\\[
\det\\!
\begin{pmatrix}
a_x - d_x & a_y - d_y & (a_x - d_x)^2 + (a_y - d_y)^2 \\\\
b_x - d_x & b_y - d_y & (b_x - d_x)^2 + (b_y - d_y)^2 \\\\
c_x - d_x & c_y - d_y & (c_x - d_x)^2 + (c_y - d_y)^2
\end{pmatrix} > 0
\\]

Si le déterminant vaut zéro, les 4 points sont cocirculaires (cas dégénéré) ; s'il est négatif, \\(D\\) est strictement à l'extérieur.

C'est ce prédicat (sans normalisation, calculé en `f64`) que pyrucast utilise dans Bowyer-Watson.

## Bowyer-Watson : insertion incrémentale

Pour insérer un nouveau point \\(p\\) dans une triangulation de Delaunay :

1. On identifie l'ensemble **bad** des triangles dont le cercle circonscrit contient \\(p\\) (déterminant ci-dessus > 0).
2. Leur union forme un polygone **étoilé** autour de \\(p\\) (théorème de Bowyer 1981 et Watson 1981).
3. On retire ces triangles ; le bord de la cavité est un polygone simple.
4. On retriangule en éventail depuis \\(p\\) : pour chaque arête \\((u, v)\\) de la cavité, on crée le triangle \\((u, v, p)\\).

L'invariant Delaunay est préservé par construction : les nouveaux triangles ne peuvent pas avoir d'autre point dans leur cercle circonscrit, sinon ce triangle aurait été dans la cavité.

**Initialisation** : pyrucast utilise un **super-triangle** englobant largement la AABB du nuage de points, ce qui garantit que tout point inséré tombera dans au moins un bad triangle. Les triangles touchant le super-triangle sont retirés à la fin.

## Triangulation contrainte (CDT) : forcer des arêtes

Une **CDT** étend Delaunay avec des arêtes imposées (typiquement les arêtes du contour à mailler). La propriété du cercle vide n'est plus exigée à travers ces arêtes contraintes.

Pour forcer une arête \\((a, b)\\) absente :

1. Identifier tous les triangles **strictement traversés** par le segment \\((a, b)\\). On suit une « marche » : on part d'un triangle contenant \\(a\\) ; on traverse en suivant les arêtes coupées.
2. Retirer ces triangles ; le bord de la cavité forme deux **polygones simples**, l'un à gauche, l'autre à droite de l'arête \\((a, b)\\). On les sépare via le signe du produit en croix.
3. Retrianguler chaque polygone par ear clipping.

## Identification intérieur / extérieur : flood-fill par parité

**Théorème de Jordan** : tout polygone simple fermé partage le plan en deux composantes connexes — un intérieur borné et un extérieur non borné. Chaque traversée du polygone bascule de l'une à l'autre.

Pour un domaine **à trous**, on étend par parité :

| Nombre de contraintes traversées depuis l'extérieur | Statut |
|---:|---|
| 0 | extérieur (hors du contour englobant) |
| 1 | intérieur du domaine maillé |
| 2 | dans un trou |
| 3 | îlot dans un trou |
| ... | alterne |

Le **flood-fill par parité** :

1. Tout triangle adjacent au super-triangle est étiqueté « extérieur ».
2. On propage en BFS le long des voisins :
   - Si l'arête entre le triangle courant et son voisin est contrainte ⇒ on **inverse** la couleur.
   - Sinon ⇒ on **conserve**.
3. À la fin, on garde uniquement les triangles « intérieurs ».

Cela fonctionne pour n'importe quel nombre de trous emboîtés, sans tester explicitement la containment.

## Raffinement de Ruppert : points Steiner

L'algorithme de **Ruppert (1995)** raffine une CDT pour satisfaire deux critères de qualité utilisateur :

- une **longueur d'arête maximale** \\(h_{\max}\\) ;
- un **angle minimum** \\(\alpha_{\min}\\), équivalent à un critère sur le rapport circumrayon / arête plus courte :

  \\[
  \frac{r}{L_{\min}} \le \frac{1}{2 \sin \alpha_{\min}}
  \\]

### Encroachment

Une arête contrainte \\(AB\\) est dite **encroachée** par un point \\(p\\) si \\(p\\) est strictement à l'intérieur du **disque diamétral** de \\(AB\\) — c'est-à-dire le disque de centre \\(M = (A + B)/2\\) et de rayon \\(\|AB\|/2\\). Condition :

\\[
\| p - M \|^2 < \frac{\| A - B \|^2}{4}
\\]

### Boucle de Ruppert

```text
boucle:
    Si une arête contrainte AB est encroachée par un sommet :
        couper AB en son milieu M (M devient un Steiner, AB est remplacé par AM + MB).
        continuer.

    Sinon, chercher un triangle "mauvais" :
        - longueur d'arête > h_max, OU
        - rapport r/L_min > 1/(2 sin α_min) (triangle "skinny")
    Si aucun : terminé.

    Sinon, soit C le centre du cercle circonscrit du mauvais triangle.
    Si C encroache une arête contrainte :
        couper cette arête en son milieu (Steiner) — retour au début de boucle.
    Sinon :
        insérer C par Bowyer-Watson contraint.
```

L'insertion **contrainte** (Bowyer-Watson modifié) propage la cavité en BFS depuis le triangle contenant \\(C\\), **sans jamais franchir d'arête contrainte**. Cela préserve toutes les contraintes initiales et les nouvelles (milieux d'arêtes).

> **Contour figé dans pyrucast.** L'algorithme de Ruppert ci-dessus coupe une arête de bord encroachée en son milieu (les deux étapes « couper AB en son milieu »). `triangulate_surface` **ne le fait pas** : le contour d'entrée doit être conservé à l'identique (mêmes `NodeId`, mêmes positions). Un point ou un circoncentre qui encroache une arête contrainte est donc simplement **abandonné** au lieu de la bissecter. Le raffinement ne pose que des Steiner **intérieurs** ; un bord finement maillé se prépare en amont (`mesher.line/arc/circle`).

### Convergence

La preuve de **terminaison** de Ruppert (renforcée par Shewchuk en 1996) tient pour

\\[
\alpha_{\min} \le 20.7^{\circ} \approx \arcsin\\!\frac{1}{2\sqrt{2}}
\\]

Au-delà de cette borne, certaines configurations très étirées peuvent conduire à des cycles d'insertion (chaque point inséré crée un nouveau triangle « skinny »). pyrucast plafonne le nombre total d'insertions à \\( 50 \cdot n_\text{contour} + 1000 \\) ; si la limite est atteinte, la fonction renvoie une erreur explicite plutôt que de boucler indéfiniment.

### Garanties asymptotiques

Pour des entrées « raisonnables » (arêtes du contour formant des angles entre elles ≥ 60°), le maillage final a :

- toutes les arêtes \\(\le h_{\max}\\) ;
- tous les angles \\(\ge \alpha_{\min}\\) ;
- une taille proche du minimum (constante de l'ordre du logarithme du ratio AABB / arête la plus courte de l'entrée).

## Références

- **Bowyer, A.** *Computing Dirichlet tessellations.* Computer Journal 24(2), 1981.
- **Watson, D. F.** *Computing the n-dimensional Delaunay tessellation with application to Voronoi polytopes.* Computer Journal 24(2), 1981.
- **Newell, M. E.** *The utilization of procedure models in digital image synthesis.* PhD thesis, Univ. of Utah, 1975.
- **Ruppert, J.** *A Delaunay refinement algorithm for quality 2-dimensional mesh generation.* J. Algorithms 18(3), 1995.
- **Shewchuk, J. R.** *Delaunay refinement mesh generation.* PhD thesis, CMU, 1997.
- **Shewchuk, J. R.** *Triangle: Engineering a 2D quality mesh generator and Delaunay triangulator.* WACG, 1996.
