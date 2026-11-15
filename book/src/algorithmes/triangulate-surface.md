# Maillage d'un contour fermé : `triangulate_surface`

{{#pyalgo mesh.triangulate_surface}}

`triangulate_surface(contour, element_type, size=None)` remplit l'intérieur
d'un contour par **triangulation de Delaunay contrainte (CDT)** puis
**raffinement de Ruppert** à taille de maille cible, en créant les nœuds
internes nécessaires. C'est l'équivalent de l'opérateur Cast3M `SURF`. Le
mailleur est rapide (≈ 3·10⁵ mailles/s) et gère nativement les **trous** et
**plusieurs domaines disjoints** en une passe.

Le **contour est figé** : le maillage produit réutilise exactement les nœuds
d'entrée (mêmes identifiants, mêmes positions) et **n'ajoute aucun nœud sur une
arête du contour**. Le raffinement n'insère donc que des nœuds *intérieurs* ;
pour un bord plus fin, discrétisez le contour en amont (`mesh.line(a, b,
15)`, `mesh.arc(...)`, `mesh.circle(...)`).

`contour` est un `Mesh` contenant **une ou plusieurs boucles SEG2** fermées ;
la configuration peut être en dimension **2** (cas direct) ou une boucle
**plane en 3D** (voir *Contrôle de planéité* plus bas). `element_type` vaut
`"TRI3"` ou `"QUA4"` ; `size` fixe la longueur d'arête visée (par défaut :
longueur moyenne des segments de bord de chaque domaine).

Les fondements mathématiques (aire signée, Newell, Delaunay / Bowyer-Watson,
CDT, Ruppert) sont rassemblés dans
[Triangulation : briques mathématiques](../triangulation.md). Cette page-ci
décrit le **comportement** de `triangulate_surface`.

## Convention d'orientation (à la charge de l'appelant)

`triangulate_surface` s'appuie sur l'**orientation** des boucles fournies :

- une boucle **antihoraire** (CCW, aire signée > 0) est la **frontière
  extérieure** d'un domaine ;
- une boucle **horaire** (CW, aire signée < 0) est un **trou**, contenu dans
  une boucle extérieure ;
- plusieurs boucles CCW disjointes maillent **plusieurs domaines** en une fois.

C'est exactement l'orientation produite par [`border`](maillage-topologie.md#bord-dune-surface--border)
(extérieur CCW, trous CW), donc la sortie de `border` réalimente directement
`triangulate_surface`.

## Méthode

1. les points de bord sont insérés un à un dans une triangulation de Delaunay
   par Bowyer-Watson (super-triangle englobant) ;
2. chaque arête de boucle absente est **récupérée** (retrait du corridor +
   ear-clipping des deux polygones adjacents) puis marquée contrainte ;
3. la triangulation est **légalisée** (flips de Delaunay ne traversant aucune
   contrainte) ;
4. **excavation** : un flood-fill depuis un triangle sûrement intérieur, ne
   traversant jamais une arête contrainte, sépare l'intérieur du domaine des
   trous et des poches hors d'un bord concave ;
5. **raffinement de Ruppert** : les triangles trop plats/trop grands sont
   coupés par insertion de leur circoncentre. Le **contour restant figé**, un
   circoncentre qui *empiéterait* une arête de bord (ou tomberait hors du
   domaine) est **abandonné** plutôt que de couper cette arête — on préserve le
   contour au prix d'un triangle un peu moins bon près du bord ;
6. léger lissage laplacien, puis (pour `QUA4`) recombinaison gloutonne des
   paires de triangles.

En `QUA4` le résultat est donc **quad-dominant** : les triangles sont
recombinés par paires en quadrangles, une poignée de triangles de bord pouvant
subsister (sous-maillage `TRI3` annexe). Les éléments sont orientés **CCW**.

## Contrôle de planéité (cas 3D)

Une boucle 3D est ajustée à son plan de meilleure approximation (méthode de
Newell), maillée dans ce repère 2D local, puis relevée dans l'espace 3D. La
déviation maximale d'un nœud du contour à ce plan doit rester inférieure à
`1e-6 × diag` (`diag` = diagonale de la boîte englobante) ; au-delà,
`triangulate_surface` retourne une erreur indiquant la déviation observée et la
tolérance. Ce seuil relatif tolère le bruit numérique tout en refusant les
vrais contours gauches.

## Raffinement — convergence

Le raffinement de Ruppert garantit théoriquement sa convergence pour un angle
minimal `≤ 20.7°` (Shewchuk) ; pyrucast vise 20° et plafonne le nombre
d'insertions pour éviter les divergences (erreur explicite si la limite est
atteinte). Les nouveaux nœuds (« Steiner ») sont **strictement intérieurs** et
créés dans la `Coords` du contour, exactement comme les nœuds utilisateur ;
aucun n'est posé sur le contour, qui reste figé. Cette contrainte peut laisser
subsister, contre un bord grossièrement discrétisé, un triangle plus plat que
l'angle visé : affiner alors le contour d'entrée plutôt que la taille cible.

## Exemple Python

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-triangulate_surface}}
```

## Refus final

Comme tous les mailleurs, `triangulate_surface` mesure ses mailles avant de les
rendre et **sort en erreur** plutôt que d'en livrer une retournée ou plate — un
jacobien négatif ou nul n'est intégrable par aucun code éléments finis. Il le
gagne par construction (le flood-fill garde les triangles dans le sens du
super-triangle, les bascules et le lissage refusent tout mouvement qui en
retournerait un, la recombinaison n'apparie que des quadrangles convexes), mais
la vérification est faite quand même : une garantie que rien ne contrôle n'en
est une que tant que les quatre tiennent.

## Interruption

Un maillage trop long s'**interrompt** par `Ctrl+C` : `triangulate_surface`
sonde les signaux et lève une `KeyboardInterrupt`. Côté Rust,
`triangulate_surface_cancellable(contour, type, size, &cancel)` accepte un
jeton d'interruption (timeout, drapeau partagé…) — voir
[Interrompre une fonction](../developper/interrompre-une-fonction.md).

## Limitations actuelles

- **Taille uniforme** : pas encore de champ de densité variable par nœud
  (un `size` façon `CHPO1`).
- L'orientation est à fournir par l'appelant ; une boucle mal orientée mène à
  une erreur (aucune boucle extérieure) ou à un domaine inattendu.
- Les boucles doivent être deux à deux disjointes (pas de trous emboîtés, pas
  de croisements).

Côté Rust, `ops::mesh::triangulate_surface(&contour, ElementType::TRI3, Some(0.5))`.
Le cœur (CDT + raffinement) opère sur de simples `Vec<Point2>` sans toucher à
la `Coords` ; lissage et recombinaison QUA4 sont parallélisés (`rayon`). Le module
`pyrucast::ops::mesh::triangulation` regroupe par ailleurs les briques
géométriques réutilisables indépendamment du système `Mesh`
(voir [Triangulation](../triangulation.md)).
