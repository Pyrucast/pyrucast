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

## Nœud le plus proche

- **`nearest_node(mesh, point)`** — le **nœud** du maillage le plus proche
  (distance euclidienne) de `point`. Question purement nodale, complémentaire de
  `locate_points` (qui, elle, renvoie la *maille* contenant le point) : seuls les
  nœuds effectivement référencés par une maille sont candidats, et les ex-æquo
  sont départagés par le plus petit identifiant (résultat déterministe). Pratique
  pour cibler un nœud où poser une condition aux limites ou lire un résultat quand
  on connaît sa position approximative mais pas son id. Exposée côté Python
  (`mesh.nearest_node([x, y])`).

Sont encore prévus, au fil des besoins : boîtes englobantes (AABB), centroïdes,
aires/volumes, métriques de **qualité** d'élément. Les briques de Jacobien
existantes vivent aujourd'hui sur le
[`SubFiniteElementSpace`](../fe-space.md) (`jacobian`, `det_jacobian`,
`dn_dx`).
