# Coordonnées (Coords)

Un `Coords` héberge **une ou plusieurs configurations** (jeux de coordonnées)
pour le même ensemble de nœuds, en dimension fixée. C'est le premier objet du
modèle pyrucast — tous les autres (Mesh, NodeField, FE space…) viennent s'y
greffer. L'**accesseur** utilisateur d'un nœud, le [`Node`](node.md), fait
l'objet du chapitre suivant.

## Repère de révolution

Un `Coords` déclare aussi **comment lire ses coordonnées** : cartésien par
défaut, ou **axisymétrique** — le plan méridien \\( (r, z) \\) d'un solide de
révolution, avec \\( x = r \ge 0 \\) (rayon) et \\( y = z \\) (axe). C'est
l'équivalent du `OPTI MODE AXIS` de Cast3M, et sa place est bien la géométrie :
le repère change la **mesure d'intégration** elle-même,

\\[
d\Omega = 2\pi r \\, |J| \\, d\xi,
\\]

donc rigidité, masse, conductivité, flux réparti, volumes et forces internes
d'un coup — sur le corps comme sur ses bords. Le facteur \\( 2\pi \\) est celui
de l'**anneau complet** : les masses, volumes et résultantes nodales sont ceux de
la pièce de révolution entière.

```rust,ignore
use pyrucast::coords::Coords;

// Cartésien (par défaut) : dim libre.
let plan = Coords::new(2).unwrap();
assert!(!plan.is_axisymmetric());

// Révolution : la dimension vaut nécessairement 2, donc pas d'argument.
let axi = Coords::axisymmetric().unwrap();
assert_eq!(axi.dim(), 2);
assert!(axi.is_axisymmetric());
```

```python
{{#include ../../tests/python/test_doc_coords.py:axisymetrique}}
```

Un rayon négatif est refusé à l'ajout (et au `set_position`) plutôt que de
ressortir en `|J|` négatif au fond d'une intégrale. Tout espace éléments finis
bâti sur ces `Coords` hérite du repère, si bien qu'un corps et son bord ne
peuvent pas diverger.

Côté **mécanique**, l'axisymétrie ajoute la déformation orthoradiale
\\( \varepsilon_{\theta\theta} = u_r/r \\), qui relève du modèle et non de la
géométrie : voir [Élasticité linéaire](mecanique/elasticite.md#axisymétrie). La
**thermique** n'a rien à changer.

Côté **visualisation**, un tracé axisymétrique montre par défaut la section
méridienne ; l'option `revolve` la balaie pour dessiner le corps de révolution
lui-même — voir
[Visualisation](visualization.md#axisymétrie--section-méridienne-ou-corps-de-révolution).

## Identité d'un nœud

Chaque nœud créé reçoit un identifiant interne **stable** (`NodeId`), unique
pour toute la vie du `Coords` : **aucun id n'est jamais réutilisé**, même après
ramassage par le GC. C'est ce qui permet aux maillages et champs de référencer
un nœud par son id sans s'inquiéter de la stabilité.

## Politique de suppression : pas de suppression directe

**Il n'existe aucune méthode `remove_node`.** Un nœud référencé est protégé.
Seul le ramasse-miettes `Coords::gc()` retire les nœuds dont le refcount
**interne** est tombé à 0.

```rust,ignore
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;

let coords = Handle::new(Coords::new(2).unwrap());
// add_node initialise refcount = 1 ; sans décrément, le nœud est protégé.
let id = coords.write().add_node(&[0.0, 0.0]).unwrap();
assert_eq!(coords.write().gc(), 0);

// Après décrément, gc ramasse.
coords.write().decref(id).unwrap();
assert_eq!(coords.write().gc(), 1);
```

## Modèle de refcount à deux niveaux

```text
        Handle<Coords>             ◀── refcount de l'Arc
                │                       (le Coords est-il vivant ?)
                ▼
        ┌──────────────────┐
        │      Coords      │
        └──────────────────┘
                │
                │ refcount par nœud
                ▼                   ◀── refcount sur chaque NodeId interne
        NodeId(0)  NodeId(1)  …        (le nœud est-il vivant ?)
```

Les deux niveaux sont indépendants :

- tant qu'un `Handle<Coords>` existe, le `Coords` reste en mémoire ;
- tant qu'au moins un [`Node`](node.md) (ou un objet aval comme Mesh / Field
  via `incref`/`decref`) référence un `NodeId`, ce nœud est protégé du GC.

Le détail du niveau objet est dans [Modèle mémoire](memory-model.md) ; le
niveau nœud, dans [Nœud](node.md).

## Plusieurs configurations

Utile pour basculer entre référence / déformée / prédite. La configuration
active est désignée par index ; lire les coordonnées d'un nœud (`node.position()`
en Python, `Coords::position` côté Rust) renvoie celles de la configuration
active. `add_config(name)` clone la configuration active sous un nouveau nom.

Rust :

```rust,ignore
let c2 = coords.write().add_config("deformed");
coords.write().select(c2).unwrap();
// les `set_position` suivants modifient désormais la configuration "deformed".
```

Python :

```python
{{#include ../../tests/python/test_doc_coords.py:configurations}}
```

## Pourquoi plusieurs configurations dans un `Coords` plutôt que plusieurs `Coords` ?

cast3m suit historiquement la convention inverse : un objet de coordonnées =
une seule configuration, et on multiplie les objets. Les deux modèles ont des
compromis distincts.

| Modèle | Avantages | Limites |
|---|---|---|
| **Plusieurs configurations dans un `Coords`** *(pyrucast actuel)* | `NodeId 42` désigne le **même nœud physique** dans toutes les configurations. Les maillages et champs restent valides quelle que soit la configuration active (référence / déformée / prédite). Topologie, refcount, permutation **mutualisés**. `select(config)` est un simple changement d'index — pas de remapping aval. | Toutes les configurations ont le **même cardinal** de nœuds. Pas de configuration "partielle" ne couvrant qu'une portion du domaine. |
| **Un `Coords` par configuration** *(modèle cast3m historique)* | Chaque objet est **autonome** : sérialisation et GC indépendants. Permet des jeux de tailles différentes (sous-problèmes, maillages adaptés). | `NodeId 42` dans `coords_A` ≠ `NodeId 42` dans `coords_B` : tout maillage ou champ référençant plusieurs `Coords` doit porter une **table de correspondance** explicite. Source classique de bugs (« nœud copié au lieu de partagé »). Ajouter un nœud « à tous les `Coords` équivalents » est une opération transverse non triviale. |

Rien n'énumère les `Coords` vivants — chacun n'est atteignable que par les
handles qui le désignent (cf. [Modèle mémoire](memory-model.md)) : propager un
`add_node` « à tous les `Coords` équivalents » n'aurait de toute façon aucun
point d'entrée.

**Choix pyrucast** : un seul jeu d'identités par domaine géométrique, plusieurs
configurations pour les variantes (référence / déformée / prédite). Des
`Coords` distincts restent prévus pour les **vrais domaines indépendants**
(sous-domaines, maillages adaptés de plus haute densité).

## Permutation solveur

Une permutation optionnelle (`Vec<u32>`, longueur = `capacity`) sépare
l'**ordre solveur** de l'**identité** : `permutation[node_id]` donne l'ordre
solveur associé. Elle est posée par l'appelant aujourd'hui ; une renumérotation
réduisant la bande/profil (Cuthill–McKee) la calculera. L'identité (`NodeId`)
n'est jamais modifiée, dans les deux cas.

Rust :

```rust,ignore
use pyrucast::coords::Coords;
use pyrucast::handle::Handle;

let coords = Handle::new(Coords::new(2).unwrap());
// Trois nœuds créés ; ids = 0, 1, 2.
coords.write().add_node(&[0.0, 0.0]).unwrap();
coords.write().add_node(&[1.0, 0.0]).unwrap();
coords.write().add_node(&[0.5, 1.0]).unwrap();

// Permutation posée à la main (le calcul automatique reste à écrire).
coords.write().set_permutation(vec![2, 0, 1]).unwrap();
// permutation[0] = 2 : le nœud d'id 0 est en position solveur 2.
println!("{:?}", coords.read().permutation());

// Retour à l'identité.
coords.write().clear_permutation();
```

Python :

```python
{{#include ../../tests/python/test_doc_coords.py:permutation}}
```

## API Python

```python
{{#include ../../tests/python/test_doc_coords.py:cycle_de_vie}}
```

Méthodes d'inspection utiles : `node_count()` (nœuds vivants), `capacity()`
(slots alloués, vivants + non encore collectés), `is_alive(id)` et
`refcount(id)` — qui prennent un **id brut** (pas un `Node`) : un `Node`
portant un refcount ne pourrait jamais être observé mort. `acquire(id)` rend un
`Node` supplémentaire pour un id existant (refcount += 1). `dump()` imprime le
contenu intégral (coordonnées, configurations) sur la sortie standard.
