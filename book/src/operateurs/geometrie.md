# Opérateurs géométriques

Le module `ops::geom` est **réservé** aux **mesures géométriques** : tout ce qui
prend un [`Mesh`](../mesh.md) / `SubMesh` (et éventuellement un champ de
coordonnées) et renvoie un scalaire ou une grandeur géométrique dérivée.

Sont prévus, au fil des besoins :

- boîtes englobantes (AABB) ;
- centres de gravité (centroïdes) ;
- aires / volumes ;
- helpers de Jacobien, normales de face ;
- métriques de **qualité** d'élément (allongement, angles…).

**Aucune fonction n'est encore exposée** — le module est vide pour l'instant.
Les briques de Jacobien et de mesure existantes vivent aujourd'hui sur le
[`SubFiniteElementSpace`](../fe-space.md) (`jacobian`, `det_jacobian`,
`dn_dx`) ; les mesures purement géométriques sur le maillage (sans formulation
EF) atterriront ici quand un opérateur concret en aura besoin.

> Cette page est conservée pour refléter fidèlement la **structure thématique**
> des modules `ops/` ; elle sera étoffée dès que `ops::geom` recevra ses
> premières fonctions.
