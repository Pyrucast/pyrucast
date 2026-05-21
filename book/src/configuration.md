# Configuration

Une `Configuration` héberge **un ou plusieurs jeux de coordonnées** pour le même ensemble de nœuds, en dimension fixée. C'est le premier objet du modèle pyrucast — tous les autres objets (Mesh, NodeField, FE space…) viendront s'y greffer.

## Identité d'un nœud

Chaque nœud créé reçoit un identifiant interne **stable** (`NodeId`), unique pour toute la vie de la `Configuration` : **aucun id n'est jamais réutilisé**, même après ramassage par le GC. C'est ce qui permet aux maillages et champs (à venir) de référencer un nœud par son id sans s'inquiéter de la stabilité.

## Politique de suppression : pas de suppression directe

**Il n'existe aucune méthode `remove_node`.** Un nœud référencé est protégé. Seul le ramasse-miettes `Configuration::gc()` retire les nœuds dont le refcount **interne** est tombé à 0.

```rust,ignore
use pyrucast::configuration::Configuration;
use pyrucast::store::{insert, with_mut};

let cfg = insert(Configuration::new(2).unwrap());
// add_node initialise refcount = 1 ; sans décrément, le nœud est protégé.
let id = with_mut(&cfg, |c| c.add_node(&[0.0, 0.0])).unwrap().unwrap();
with_mut(&cfg, |c| assert_eq!(c.gc(), 0)).unwrap();

// Après décrément, gc ramasse.
with_mut(&cfg, |c| c.decref(id)).unwrap().unwrap();
with_mut(&cfg, |c| assert_eq!(c.gc(), 1)).unwrap();
```

## `Node` : interface RAII

Le `Node` est l'**accesseur utilisateur** d'un nœud. Il porte un handle vers la `Configuration` et un `NodeId`, et maintient le refcount **automatiquement** :

- `Clone` incrémente,
- `Drop` décrémente.

```rust,ignore
use pyrucast::configuration::Configuration;
use pyrucast::node::Node;
use pyrucast::store::{insert, with_mut};

let cfg = insert(Configuration::new(2).unwrap());
let n = Node::create_in(cfg.clone(), &[1.0, 2.0]).unwrap();
let m = n.clone();       // refcount = 2
drop(n);                 // refcount = 1
drop(m);                 // refcount = 0
with_mut(&cfg, |c| c.gc()).unwrap();  // collecte
```

Le code interne peut toujours manipuler directement les `NodeId` sans passer par `Node`, mais perd alors la protection automatique du GC : il doit appeler manuellement `Configuration::incref` / `Configuration::decref` (utilisé par les maillages et les champs à venir).

## Modèle de refcount à deux niveaux

```text
        Handle<Configuration>      ◀── refcount sur le slot du store global
                │                       (la Configuration est-elle vivante ?)
                ▼
        ┌──────────────────┐
        │   Configuration  │
        └──────────────────┘
                │
                │ refcount par nœud
                ▼                   ◀── refcount sur chaque NodeId interne
        NodeId(0)  NodeId(1)  …        (le nœud est-il vivant ?)
```

Les deux niveaux sont indépendants :

- tant qu'un `Handle<Configuration>` existe, la `Configuration` reste en mémoire ;
- tant qu'au moins un `Node` (ou un objet aval comme Mesh/Field via `incref`/`decref`) référence un `NodeId`, ce nœud est protégé du GC.

## Plusieurs jeux de coordonnées

Utile pour basculer entre référence / déformée / prédite. Le jeu actif est désigné par index ; `coord(id)` lit le jeu actif. `add_coord_set(name)` clone le jeu actif sous un nouveau nom.

```rust,ignore
let s2 = with_mut(&cfg, |c| c.add_coord_set("deformed")).unwrap();
with_mut(&cfg, |c| c.switch_to(s2)).unwrap().unwrap();
// les `set_coord` suivants modifient désormais le jeu "deformed".
```

## Pourquoi plusieurs jeux dans une `Configuration` plutôt que plusieurs `Configuration` ?

cast3m suit historiquement la convention inverse : une `Configuration` = un seul jeu de coordonnées, et on multiplie les objets. Les deux modèles ont des compromis distincts.

| Modèle | Avantages | Limites |
|---|---|---|
| **Plusieurs jeux dans une `Configuration`** *(pyrucast actuel)* | `NodeId 42` désigne le **même nœud physique** dans tous les jeux. Les maillages et champs restent valides quel que soit le jeu actif (référence / déformée / prédite). Topologie, refcount, permutation **mutualisés**. `switch_to(set)` est un simple changement d'index — pas de remapping aval. | Tous les jeux ont le **même cardinal** de nœuds. Pas de jeu "partiel" ne couvrant qu'une portion du domaine. |
| **Une `Configuration` par jeu** *(modèle cast3m historique)* | Chaque objet est **autonome** : swap, sérialisation, GC indépendants. Permet des jeux de tailles différentes (sous-problèmes, maillages adaptés). | `NodeId 42` dans `cfg_A` ≠ `NodeId 42` dans `cfg_B` : tout maillage ou champ référençant plusieurs configs doit porter une **table de correspondance** explicite. Source classique de bugs (« nœud copié au lieu de partagé »). Ajouter un nœud « à toutes les configs équivalentes » est une opération transverse non triviale. |

Le store sait mécaniquement énumérer toutes les `Configuration` résidentes (registre indexé par `TypeId`), mais propager un `add_node` à toutes deviendrait coûteux : il faudrait recharger en mémoire les slots `OnDisk`, ce qui annulerait un des bénéfices du swap (cf. [Modèle mémoire](memory-model.md)).

**Choix pyrucast** : un seul jeu d'identités par domaine géométrique, plusieurs jeux de coordonnées pour les variantes (référence / déformée / prédite). Des `Configuration` distinctes restent prévues pour les **vrais domaines indépendants** (sous-domaines, maillages adaptés de plus haute densité).

## Permutation solveur

Une permutation optionnelle (`Vec<u32>`, longueur = `capacity`) sépare l'**ordre solveur** de l'**identité** : `permutation[node_id]` donne l'ordre solveur associé. Phase 4 (renumérotation Cuthill–McKee) la calculera pour réduire la bande/profil. L'identité (`NodeId`) n'est jamais modifiée.

## API Python

```python
import pyrucast

c = pyrucast.Configuration(dim=2)
n = c.add_node([0.0, 0.0])      # n est un pyrucast.Node ; refcount = 1
m = c.add_node([1.0, 0.0])

print(c)                         # Configuration: dim=2, sets=1 (active="default"), nodes=2 ...
n.set_coord([0.5, 0.5])

# GC ne touche pas tant qu'au moins un Node Python existe.
assert c.gc() == 0

# del + collect force le Drop côté Rust et libère le refcount.
import gc as pygc
del n
pygc.collect()
assert c.gc() == 1
```
