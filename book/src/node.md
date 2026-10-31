# Nœud (Node)

Un `Node` désigne un nœud d'un [`Coords`](coords.md). Il ne stocke pas de
coordonnées : il porte son `Coords` et l'identifiant du nœud, et lit tout dans
le `Coords` — les coordonnées qu'il rend sont donc celles de la configuration
active. C'est l'objet qu'on passe partout où une fonction attend un nœud :
`pyrucast.mesh.line(a, b, 10)`, `pyrucast.mesh.arc(…)`, `pyrucast.mesh.circle(…)`.

## Créer et retrouver un nœud

Un `Node` naît toujours d'un `Coords` :

- `coords.add_node([x, y, …])` crée le nœud et rend son `Node` ;
- `coords.acquire(id)` rend un `Node` pour un nœud qui existe déjà — un nœud
  posé par un mailleur, par exemple, dont on connaît l'identifiant.

## Interface

- `id` — l'identifiant entier, stable dans son `Coords` ;
- `position()` — ses coordonnées dans la configuration active ;
- `set_position([x, y, …])` — les réécrit, dans la configuration active ;
- `coords()` — le `Coords` auquel il appartient ;
- `node | node` — un `Mesh` POI1 sur les deux nœuds (la même union `|` que les
  agrégats, cf. [Agrégat](aggregate.md)), et `mesh | node` ajoute un point à un
  `Mesh` POI1 d'une seule zone ;
- `print(n)`, `repr(n)` et `n.dump()` — les
  [trois niveaux de listing](aspect-informatique.md#trois-niveaux-de-listing).

```python
{{#include ../../tests/python/test_doc_conteneurs.py:node_api}}
```

## Un `Node` compte comme une référence

Chaque `Node` compte comme une référence à son nœud, celle que
`coords.refcount(id)` additionne avec celles des maillages et des champs :
`add_node` et `acquire` en ajoutent une, et le `Node` la rend quand Python le
détruit. Une seconde variable sur le même objet (`b = a`) ne compte pas deux
fois : c'est le même `Node`.

C'est ce qui le distingue d'un simple identifiant. Un entier ne dit pas à quel
`Coords` il appartient et ne compte pour rien ; le `Node` porte son `Coords`,
lit ses coordonnées dans la bonne configuration, et le nœud le compte parmi
ses références tant qu'il existe. Porter son `Coords` lui permet aussi d'être
refusé là où il n'a pas sa place : une maille, un champ, une matrice posés sur
un autre `Coords` lèvent une erreur plutôt que de prendre le nœud de ce
`Coords` qui porte le même numéro.

## Côté Rust

En Rust, un `Node` est une paire `(Handle<Coords>, NodeId)` qui tient son
compte par RAII : `Clone` ajoute une référence dans le `Coords`, `Drop` la rend.
Il se crée par `Node::create_in(coords, &[x, y, …])`, et
`Node::acquire(coords, id)` en rend un de plus pour un identifiant existant —
le verbe que Python porte sur le `Coords`, Rust le porte sur le nœud. Il expose
`id()`, `position()`, `set_position(&[…])` et `coords()`, qui rend le
`Handle<Coords>` porté.

```rust,ignore
{{#include ../../tests/doc_conteneurs.rs:noeud}}
```

Le code interne peut manipuler des `NodeId` sans passer par `Node`, à condition
de tenir lui-même le compte avec `Coords::incref` / `Coords::decref` : c'est ce
que font les maillages et les champs. Le détail est dans
[Modèle mémoire](memory-model.md).
