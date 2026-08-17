# Nœud (Node)

Le `Node` est l'**accesseur utilisateur** d'un nœud d'un [`Coords`](coords.md).
Il ne stocke pas de coordonnées : il porte un `Handle<Coords>` et un `NodeId`,
et délègue tout au `Coords` (les coordonnées lues dépendent donc de la
configuration active). C'est l'objet qu'on passe partout où une API attend un
nœud.

## Principe : un identifiant qui se protège

Un `Node` est conceptuellement une paire `(Coords, NodeId)`, mais avec une
propriété de plus : il **maintient le refcount interne automatiquement**, par
RAII.

- `Clone` **incrémente** le refcount du nœud dans le `Coords` ;
- `Drop` le **décrémente**.

Tant qu'un `Node` (ou un objet aval — un `SubMesh`, un champ — qui a fait son
propre `incref`) référence un `NodeId`, ce nœud est protégé du ramasse-miettes
`Coords::gc()`. C'est le **niveau interne** du refcount à deux niveaux décrit
dans [Coordonnées](coords.md).

```rust,ignore
use pyrucast::atoms::Node;
use pyrucast::coords::Coords;
use pyrucast::store::Handle;

let coords = Handle::new(Coords::new(2).unwrap());
let n = Node::create_in(coords.clone(), &[1.0, 2.0]).unwrap();
let m = n.clone();             // refcount = 2
drop(n);                       // refcount = 1
drop(m);                       // refcount = 0
coords.write().gc();  // collecte
```

Le code interne peut toujours manipuler directement les `NodeId` sans passer
par `Node`, mais perd alors la protection automatique : il doit appeler
manuellement `Coords::incref` / `Coords::decref` (c'est ce que font les
maillages et les champs).

## Création

Un `Node` ne se construit jamais « dans le vide » : il naît d'un `Coords`.

- **Rust** : `Node::create_in(coords_handle, &[x, y, …])` crée le nœud et rend
  le `Node` (refcount = 1). Pour obtenir un `Node` supplémentaire sur un id
  déjà existant, `Coords::acquire(id)`.
- **Python** : `coords.add_node([x, y, …])` renvoie directement un
  `pyrucast.Node`. `coords.acquire(id)` rend un accesseur de plus.

## Interface

Côté **Python**, le `Node` expose :

- la propriété `id` — l'identifiant entier stable dans son `Coords` ;
- `position()` — ses coordonnées dans la configuration active ;
- `set_position([x, y, …])` — réécrit ses coordonnées (dans la configuration
  active) ;
- `coords()` — le `Coords` auquel il appartient ; filet de secours quand la
  poignée a été lâchée côté Python, comme `Mesh.coords()` ;
- l'union `node | node` → un `Mesh` POI1 unitaire sur les deux nœuds (la même
  union `|` que les agrégats, cf. [Agrégat](aggregate.md)) ; et `mesh | node`
  ajoute un point à un `Mesh` POI1 unitaire ;
- les vues `repr` / `str` et `dump()`.

```python
import pyrucast

c = pyrucast.Coords(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])

print(a.id)  # 0
print(a.position())  # [0.0, 0.0]
a.set_position([0.5, 0.5])

# Union de nœuds → maillage POI1 (deux points).
poi = a | b
print(poi)  # Mesh: 1 submesh(es), 2 cell(s) total
```

Côté **Rust**, `Node` expose `id()`, `position()`, `set_position(&[…])`,
`coords()` (le `Handle<Coords>` porté), plus `Clone`/`Drop` qui gèrent le
refcount comme décrit ci-dessus.

## Pourquoi un accesseur protecteur plutôt qu'un simple id ?

Manipuler des `NodeId` bruts est possible mais dangereux : rien n'empêche un
nœud d'être ramassé pendant qu'on tient encore son id. Le `Node` rend la
protection **automatique et locale** — exactement comme un `Rc`/`Arc` le ferait
pour un objet du heap, mais ici au niveau du nœud interne d'un `Coords`. Une
alternative (modèle « cast3m pur » où un nœud n'existe qu'au sein d'un
maillage) est discutée dans [Modèle mémoire](memory-model.md) ; pyrucast
conserve pour l'instant le `Node` protecteur.
