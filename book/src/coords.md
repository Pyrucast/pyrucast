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
d\Omega = 2\pi r \, |J| \, d\xi,
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
import pyrucast

c = pyrucast.Coords.axisymmetric()
assert c.dim == 2 and c.is_axisymmetric
c.add_node([1.0, 0.0])  # r = 1, z = 0
c.add_node([-1.0, 0.0])  # erreur : x est un rayon, il doit être ≥ 0
```

Un rayon négatif est refusé à l'ajout (et au `set_coord`) plutôt que de
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
use pyrucast::store::{insert, write};

let coords = insert(Coords::new(2).unwrap());
// add_node initialise refcount = 1 ; sans décrément, le nœud est protégé.
let id = write(&coords).unwrap().add_node(&[0.0, 0.0]).unwrap();
assert_eq!(write(&coords).unwrap().gc(), 0);

// Après décrément, gc ramasse.
write(&coords).unwrap().decref(id).unwrap();
assert_eq!(write(&coords).unwrap().gc(), 1);
```

## Modèle de refcount à deux niveaux

```text
        Handle<Coords>             ◀── refcount sur le slot du store global
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

Le détail du niveau slot (générations, swap, compactage) est dans
[Modèle mémoire](memory-model.md) ; le niveau nœud, dans [Nœud](node.md).

## Plusieurs configurations

Utile pour basculer entre référence / déformée / prédite. La configuration
active est désignée par index ; lire les coordonnées d'un nœud (`node.coord()`
en Python, `Coords::coord` côté Rust) renvoie celles de la configuration
active. `add_config(name)` clone la configuration active sous un nouveau nom.

Rust :

```rust,ignore
let c2 = write(&coords).unwrap().add_config("deformed");
write(&coords).unwrap().select(c2).unwrap();
// les `set_coord` suivants modifient désormais la configuration "deformed".
```

Python :

```python
import pyrucast

c = pyrucast.Coords(dim=2)
n = c.add_node([0.0, 0.0])

# Créer une deuxième configuration (clone de la configuration active).
c2 = c.add_config("deformed")
print(c.names())  # ['default', 'deformed']

# Basculer sur la configuration déformée et modifier les coordonnées.
c.select(c2)
n.set_coord([0.1, 0.05])

# Les coordonnées lues dépendent de la configuration active.
c.select(0)
print(n.coord())  # [0.0, 0.0]  — configuration de référence
c.select(c2)
print(n.coord())  # [0.1, 0.05] — configuration déformée
print(c.active)  # 1
```

## Pourquoi plusieurs configurations dans un `Coords` plutôt que plusieurs `Coords` ?

cast3m suit historiquement la convention inverse : un objet de coordonnées =
une seule configuration, et on multiplie les objets. Les deux modèles ont des
compromis distincts.

| Modèle | Avantages | Limites |
|---|---|---|
| **Plusieurs configurations dans un `Coords`** *(pyrucast actuel)* | `NodeId 42` désigne le **même nœud physique** dans toutes les configurations. Les maillages et champs restent valides quelle que soit la configuration active (référence / déformée / prédite). Topologie, refcount, permutation **mutualisés**. `select(config)` est un simple changement d'index — pas de remapping aval. | Toutes les configurations ont le **même cardinal** de nœuds. Pas de configuration "partielle" ne couvrant qu'une portion du domaine. |
| **Un `Coords` par configuration** *(modèle cast3m historique)* | Chaque objet est **autonome** : swap, sérialisation, GC indépendants. Permet des jeux de tailles différentes (sous-problèmes, maillages adaptés). | `NodeId 42` dans `coords_A` ≠ `NodeId 42` dans `coords_B` : tout maillage ou champ référençant plusieurs `Coords` doit porter une **table de correspondance** explicite. Source classique de bugs (« nœud copié au lieu de partagé »). Ajouter un nœud « à tous les `Coords` équivalents » est une opération transverse non triviale. |

Le store sait mécaniquement énumérer tous les `Coords` résidents (registre
indexé par `TypeId`), mais propager un `add_node` à tous deviendrait coûteux :
il faudrait recharger en mémoire les slots `OnDisk`, ce qui annulerait un des
bénéfices du swap (cf. [Modèle mémoire](memory-model.md)).

**Choix pyrucast** : un seul jeu d'identités par domaine géométrique, plusieurs
configurations pour les variantes (référence / déformée / prédite). Des
`Coords` distincts restent prévus pour les **vrais domaines indépendants**
(sous-domaines, maillages adaptés de plus haute densité).

## Permutation solveur

Une permutation optionnelle (`Vec<u32>`, longueur = `capacity`) sépare
l'**ordre solveur** de l'**identité** : `permutation[node_id]` donne l'ordre
solveur associé. La Phase 4 (renumérotation Cuthill–McKee) la calculera pour
réduire la bande/profil. L'identité (`NodeId`) n'est jamais modifiée.

Rust :

```rust,ignore
use pyrucast::coords::Coords;
use pyrucast::store::{insert, read, write};

let coords = insert(Coords::new(2).unwrap());
// Trois nœuds créés ; ids = 0, 1, 2.
write(&coords).unwrap().add_node(&[0.0, 0.0]).unwrap();
write(&coords).unwrap().add_node(&[1.0, 0.0]).unwrap();
write(&coords).unwrap().add_node(&[0.5, 1.0]).unwrap();

// Permutation manuelle (Cuthill–McKee automatique en Phase 4).
write(&coords).unwrap().set_permutation(vec![2, 0, 1]).unwrap();
// permutation[0] = 2 : le nœud d'id 0 est en position solveur 2.
println!("{:?}", read(&coords).unwrap().permutation());

// Retour à l'identité.
write(&coords).unwrap().clear_permutation();
```

Python :

```python
import pyrucast

c = pyrucast.Coords(dim=2)
c.add_node([0.0, 0.0])
c.add_node([1.0, 0.0])
c.add_node([0.5, 1.0])

# Affecter une permutation manuellement.
c.set_permutation([2, 0, 1])
print(c.permutation())  # [2, 0, 1]

# Retour à l'identité (None = identité).
c.clear_permutation()
print(c.permutation())  # None
```

## API Python

```python
import pyrucast

c = pyrucast.Coords(dim=2)
n = c.add_node([0.0, 0.0])  # n est un pyrucast.Node ; refcount = 1
m = c.add_node([1.0, 0.0])

print(c)  # Coords: dim=2, configs=1 (active="default"), nodes=2 ...
n.set_coord([0.5, 0.5])

# GC ne touche pas tant qu'au moins un Node Python existe.
assert c.gc() == 0

# del + collect force le Drop côté Rust et libère le refcount.
import gc as pygc

del n
pygc.collect()
assert c.gc() == 1
```

Méthodes d'inspection utiles : `node_count()` (nœuds vivants), `capacity()`
(slots alloués, vivants + non encore collectés), `is_alive(id)` et
`refcount(id)` — qui prennent un **id brut** (pas un `Node`) : un `Node`
portant un refcount ne pourrait jamais être observé mort. `acquire(id)` rend un
`Node` supplémentaire pour un id existant (refcount += 1). `dump()` imprime le
contenu intégral (coordonnées, configurations) sur la sortie standard.
