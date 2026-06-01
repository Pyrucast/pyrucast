# Champ aux nœuds (`NodeField`)

Un **`NodeField`** porte une ou plusieurs valeurs **par nœud**, sur un support défini par un sous-maillage POI1 (cf. [Maillage](mesh.md)). C'est le premier objet « non géométrique » de pyrucast et le premier qui exerce le ramasse-miettes sur des nœuds qui n'auraient plus de `Node` utilisateur.

## Support : un sous-maillage POI1

Un `SubMesh` POI1 est, par construction, **exactement une liste de nœuds** (un nœud par cellule). On l'utilise comme **sélecteur de support** pour le champ : seuls les nœuds présents dans la POI1 reçoivent une valeur.

Le support est **figé à la construction** : le `NodeField` capture la liste de `NodeId` au moment de sa création et ne suit pas les évolutions ultérieures de la POI1 originelle. Cela rend les semantiques de cohérence triviales (la longueur du tableau de valeurs ne change jamais) et évite tout couplage subtil entre l'objet POI1 et le champ.

```text
   SubMesh POI1            NodeField
   ────────────            ─────────
   cell 0 → NodeId(3)      values[0 * ncomp .. ]
   cell 1 → NodeId(7)      values[1 * ncomp .. ]
   cell 2 → NodeId(12)     values[2 * ncomp .. ]
   …                       …
```

## Composantes nommées

Chaque champ porte un ou plusieurs **noms de composantes** (`"UX"`, `"UY"`, `"T"`, `"P"`, …). Les valeurs sont rangées en **row-major** : la composante `c` du nœud `i` se trouve à l'indice `i × ncomp + c` dans le buffer plat interne.

- Au moins une composante est requise à la construction.
- Les noms doivent être uniques au sein d'un même champ.
- À la construction, **toutes les valeurs valent `0.0`**.

## Refcount sur les nœuds

À la création, le `NodeField` incrémente le refcount **interne** de chaque nœud de son support dans la `Configuration` (cf. [Configuration](configuration.md)). Son `Drop` les décrémente. Tant qu'un `NodeField` référence un nœud, le GC le protège — même si tous les `Node` utilisateurs et le `SubMesh` source ont disparu :

```text
   Configuration             ◀── refcount par NodeId
        │
        ├── Node(s) utilisateur(s)        ── chacun +1
        ├── SubMesh(s) référençant ce nœud ── chacun +1 par cellule incidente
        └── NodeField(s) sur ce support    ── chacun +1 par nœud
```

En cas d'échec partiel pendant la construction (par exemple un nœud collecté entre la lecture du SubMesh et l'incref du champ — peu probable car le SubMesh tient déjà un incref), les incréments déjà effectués sont **annulés** (rollback transactionnel).

## API Rust

```rust,ignore
use pyrucast::mesh::configuration::Configuration;
use pyrucast::mesh::element_type::ElementType;
use pyrucast::mesh::SubMesh;
use pyrucast::mesh::node::Node;
use pyrucast::containers::node_field::NodeField;
use pyrucast::store::{insert, with};

let cfg = insert(Configuration::new(2).unwrap());
let a = Node::create_in(cfg.clone(), &[0.0, 0.0]).unwrap();
let b = Node::create_in(cfg.clone(), &[1.0, 0.0]).unwrap();

// Support : SubMesh POI1 contenant a et b.
let sm = {
    let mut sm = SubMesh::new(cfg.clone(), ElementType::POI1);
    sm.add_cell(&[a.id()]).unwrap();
    sm.add_cell(&[b.id()]).unwrap();
    insert(sm)
};

// Champ de déplacement 2D : composantes UX, UY.
let mut u = NodeField::from_poi1(&sm, vec!["UX".into(), "UY".into()]).unwrap();
u.set(0, 0, 1.5).unwrap();
u.set(0, 1, -0.25).unwrap();
assert_eq!(u.get(0, 0).unwrap(), 1.5);
assert_eq!(u.get(1, 0).unwrap(), 0.0);   // valeur par défaut

// Accès par NodeId + nom de composante.
let ci_ux = u.component_index("UX").unwrap();
u.set_by_node(b.id(), ci_ux, 9.0).unwrap();
assert_eq!(u.get_by_node(b.id(), ci_ux).unwrap(), 9.0);
```

## API Python

```python
import pyrucast

c = pyrucast.Configuration(dim=2)
a = c.add_node([0.0, 0.0])
b = c.add_node([1.0, 0.0])

mesh = pyrucast.Mesh(c, "POI1")
mesh.unit().add_cell([a])
mesh.unit().add_cell([b])

u = pyrucast.NodeField(mesh, ["UX", "UY"])
u.set(0, 0, 1.5)
u.set(0, 1, -0.25)

print(u)                      # NodeField: 2 node(s), 2 component(s) [UX, UY]
print(u.node_values(0))       # [1.5, -0.25]
print(u.node_values(1))       # [0.0, 0.0]

# Accès par NodeId + nom de composante.
ci = u.component_index("UX")
u.set_by_node(b.id, ci, 9.0)
print(u.get_by_node(b.id, ci))   # 9.0
```

## Sûreté du swap

Comme `SubMesh` et `Mesh`, `NodeField` porte un effet de bord dans son `Drop` (décrément du refcount des nœuds du support). Le store traite son swap correctement :

- `swap_out` n'exécute **pas** le `Drop` de la valeur évincée (`std::mem::forget` interne) — le champ est logiquement vivant, juste relocalisé.
- Le `Drop` final s'exécute après rechargement depuis le disque si nécessaire. Comme la liste `nodes: Vec<NodeId>` est sérialisée avec le reste, le décrément exact est rejoué une seule fois sur la durée de vie de l'objet.

Voir le chapitre [Modèle mémoire](memory-model.md) pour le mécanisme général.

## Pourquoi un snapshot du support plutôt qu'un lien vivant ?

Plusieurs options ont été examinées au moment du design :

| Option | Coût | Pourquoi pas |
|---|---|---|
| Snapshot (`Vec<NodeId>` propre) — **retenu** | +1 `Vec<NodeId>` par champ | Sémantique simple, indépendance vis-à-vis des évolutions du SubMesh, refcount transactionnel par champ. |
| `Handle<SubMesh>` vivant + indexation à la volée | Lecture indirecte à chaque accès | Couplage temporel difficile à maintenir : un `add_cell` ultérieur sur le SubMesh laisserait le champ dans un état incohérent (taille du `Vec<f64>` ≠ nombre de cellules). |
| `SubMesh` figé après création d'un champ | Drapeau « frozen » + erreurs à l'ajout | Restreint l'API du SubMesh, surcharge le modèle conceptuel. |

L'option retenue rend les invariants locaux au champ et autorise le SubMesh à continuer d'évoluer indépendamment.
