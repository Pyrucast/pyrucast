# Opérateurs géométriques

Le module `ops::geom` est **réservé** aux **mesures géométriques** : tout ce qui
prend un [`Mesh`](../mesh.md) / `SubMesh` (et éventuellement un champ de
coordonnées) et renvoie un scalaire ou une grandeur géométrique dérivée.

## Localisation et projection

Deux primitives d'appariement géométrique, aujourd'hui internes (API Rust,
sous les contraintes qui les consomment) :

- **`locate_points(host, points, tol)`** — *mapping iso-paramétrique inverse* :
  pour chaque point, la maille hôte qui le **contient** et ses coordonnées de
  référence `ξ` (Newton sur `x − Σ Nᵢ(ξ)·Xᵢ`, test d'appartenance au domaine
  de référence), d'où les poids `Nᵢ(ξ)` et les nœuds de la maille. C'est la
  brique du [baignage](../contraintes/embedded.md) (`u(p) = Σᵢ Nᵢ·u(hôteᵢ)`).
- **`project_points(surface, points)`** — *projection au point le plus proche*
  sur un maillage **surfacique** (facettes de dimension `sdim−1` :
  `SEG2`/`SEG3` en 2D, `TRI*`/`QUA*` en 3D). Pour chaque point, la facette la
  plus proche, `ξ` clampé au domaine de référence (bords/coins gérés par le
  clamp), poids `Nᵢ(ξ)`, **normale** orientée et **jeu signé** `(x − p)·n`.
  C'est la brique du [contact](../contraintes/contact.md).

## Nœud le plus proche — une méthode, plus un opérateur

`mesh.nearest_node(point)` rend le **nœud** du maillage le plus proche (distance
euclidienne) de `point`. Question purement nodale, complémentaire de
`locate_points` (qui, elle, rend la *maille* contenant le point) : seuls les
nœuds effectivement référencés par une maille sont candidats, et les ex-æquo
sont départagés par le plus petit identifiant, donc le résultat est
déterministe. Pratique pour cibler un nœud où poser une condition aux limites
ou lire un résultat quand on connaît sa position approximative mais pas son id.

Elle **n'est plus un opérateur** : elle a quitté `ops::geom` pour devenir une
méthode de [`Mesh`](../mesh.md), des deux côtés. C'est ce que dit la règle —
un seul conteneur, un point pour tout autre argument, et une vue dérivée bon
marché : le cas typique de la méthode, pas de la fonction libre. Elle vivait
dans `ops::geom` par voisinage thématique avec `locate_points`, et cela créait
une asymétrie que la convention interdit : fonction libre côté Rust, méthode
côté Python.

C'est aussi la requête « un seul nœud » de la famille de **sélection par région
géométrique** — `points_in_sphere`, `points_on_plane`, `points_in_cylinder`,
`points_on_cone`, `points_on_torus`… — documentée avec les [opérateurs de
maillage](maillage.md). Ces opérateurs-là rendent toujours un maillage POI1 ;
le point le plus proche étant unique, c'est un `Node`, et il ne rentre donc pas
dans la famille.

Sont encore prévus, au fil des besoins : boîtes englobantes (AABB), centroïdes,
aires/volumes, métriques de **qualité** d'élément. Les briques de Jacobien
existantes vivent aujourd'hui sur le
[`SubFiniteElementSpace`](../fe-space.md) (`jacobian`, `det_jacobian`,
`dn_dx`).
