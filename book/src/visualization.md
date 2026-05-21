# Visualisation

La visualisation des maillages est **optionnelle** : elle est gardée par des *features* Cargo et n'est donc compilée que sur demande.

| Feature Cargo | Apport | Dépendances ajoutées |
|---|---|---|
| (aucune) | rien — bibliothèque de calcul pure | — |
| `viz` | export PNG + SVG (rendu CPU) | `plotters` |
| `viz-interactive` | + fenêtre interactive (souris) | `plotters`, `winit`, `softbuffer` |

`viz-interactive` implique `viz`. Pour un environnement sans serveur d'affichage (CI headless, conteneur), on s'arrête à `viz` : tout l'export image fonctionne.

```sh
# Bibliothèque de calcul pure (par défaut).
cargo build

# Avec export PNG/SVG.
cargo build --features viz

# Avec fenêtre interactive.
cargo build --features viz-interactive

# Côté Python (pour les tests pytest) :
maturin develop --features viz
```

## Modèle de point de vue

La caméra est décrite par une structure `View`, située sur une sphère orientée autour d'un point cible :

- `yaw` : azimut en degrés (rotation autour de l'axe Z monde) ;
- `pitch` : élévation en degrés (au-dessus du plan XY monde) ;
- `scale` : `1.0` = la bounding-box remplit l'image ; `>1` zoom, `<1` dézoom ;
- `target` : point regardé. `None` ⇒ le centre de la bounding-box de l'objet visualisé.

Préréglages disponibles :

```rust,ignore
use pyrucast::viz::View;

let _ = View::front();   // yaw=0, pitch=0      : caméra en +X
let _ = View::side();    // yaw=90, pitch=0     : caméra en +Y
let _ = View::top();     // yaw=0, pitch=90     : vue du dessus
let _ = View::iso();     // yaw=45, pitch≈35.26 : isométrique
let _ = View::default(); // = iso()
```

Convention : `yaw = pitch = 0` place la caméra en `+X`, regard vers l'origine, axe `Z` vers le haut. Le repère écran qui en résulte est `(Y, Z)`.

## Une seule fonction `plot`

Toute la sortie passe par la même méthode `plot(view, save)` exposée sur `SubMesh` et `Mesh` :

| `save` | Effet | Compilation nécessaire |
|---|---|---|
| `None` | ouvre une fenêtre interactive (souris : rotation au glisser, molette : zoom) | `viz-interactive` |
| `Some(path)` avec extension `.png` | écrit un PNG | `viz` |
| `Some(path)` avec extension `.svg` | écrit un SVG vectoriel | `viz` |

Tout autre extension est rejetée avec une erreur explicite. Le format vectoriel est ce qui rend ce socle particulièrement utile pour les figures de rapport : on conserve un trait propre quel que soit le zoom.

Exemple Rust :

```rust,ignore
use pyrucast::configuration::Configuration;
use pyrucast::element_type::ElementType;
use pyrucast::mesh::SubMesh;
use pyrucast::node::Node;
use pyrucast::store::insert;
use pyrucast::viz::View;
use std::path::Path;

let cfg = insert(Configuration::new(3).unwrap());
let a = Node::create_in(cfg.clone(), &[0.0, 0.0, 0.0]).unwrap();
let b = Node::create_in(cfg.clone(), &[1.0, 0.0, 0.0]).unwrap();
let c = Node::create_in(cfg.clone(), &[0.0, 1.0, 0.0]).unwrap();
let mut sm = SubMesh::new(cfg, ElementType::TRI3);
sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

// Export vectoriel.
sm.plot(Some(View::iso()), Some(Path::new("triangle.svg"))).unwrap();
// Fenêtre interactive (feature `viz-interactive`).
// sm.plot(None, None).unwrap();
```

Côté Python, l'API miroir prend des tuples :

```python
import pyrucast

cfg = pyrucast.Configuration(3)
a = cfg.add_node([0.0, 0.0, 0.0])
b = cfg.add_node([1.0, 0.0, 0.0])
c = cfg.add_node([0.0, 1.0, 0.0])

sm = pyrucast.SubMesh(cfg, "TRI3")
sm.add_cell([a.id, b.id, c.id])

# (yaw, pitch, scale) ; save=None ouvre la fenêtre interactive.
sm.plot(view=(45.0, 35.264, 1.0), save="triangle.svg")
```

## Couleur de face par `SubMesh`

Chaque `SubMesh` porte une propriété `face_color` (type `RgbColor`, format `(r, g, b)` sur 8 bits) utilisée par la couche viz pour remplir les facettes. Cette donnée n'a **aucun effet sur les calculs** ; elle est simplement persistée avec le maillage et consommée par `plot`. Couleur par défaut : un bleu clair (`180, 200, 230`).

Côté Rust :

```rust,ignore
use pyrucast::color::RgbColor;
use pyrucast::element_type::ElementType;
use pyrucast::mesh::SubMesh;

let mut sm = SubMesh::new(cfg, ElementType::TRI3);
sm.set_face_color(RgbColor::new(220, 60, 60));
assert_eq!(sm.face_color(), RgbColor::new(220, 60, 60));
```

Côté Python :

```python
sm = pyrucast.SubMesh(cfg, "TRI3")
sm.face_color = (220, 60, 60)
assert sm.face_color == (220, 60, 60)
```

Quand on appelle `Mesh::plot`, chaque sous-maillage est rendu avec **sa propre** `face_color`, ce qui permet de distinguer visuellement des composants regroupés dans un même maillage (par exemple : peau / cœur / interfaces).

## État actuel du support

Pour cette première itération, **seuls les éléments `TRI3` sont rendus** :

- `SubMesh::plot` sur autre chose que `TRI3` lève une erreur explicite ;
- `Mesh::plot` parcourt tous ses sous-maillages et **ignore silencieusement** ceux qui ne sont pas encore supportés (en pratique : on dessine les `TRI3`, on laisse les autres invisibles).

Les autres types (`SEG2`, `QUA4`, `TET4`, `HEX8`) seront ajoutés ensuite, sans changement d'API : l'ajout consistera à étendre l'implémentation interne dans `src/viz/mesh_draw.rs`.

## Notes techniques

- Le rendu utilise l'algorithme du **peintre** : projection 3D → 2D, tri des triangles par profondeur moyenne (du plus lointain au plus proche), puis dessin des facettes pleines suivies des arêtes noires en superposition. Coût : `O(n log n)` à chaque rafraîchissement, raisonnable jusqu'à quelques milliers de cellules. Pour des maillages industriels, prévoir un export `.vtu` vers ParaView.
- L'export reste **portable Linux ↔ Windows** : tout le rendu se fait en CPU, sans pilote GPU. Le binaire `viz-interactive` nécessite en revanche un serveur d'affichage (X11, Wayland ou Windows) à l'exécution — ce qui est attendu pour une fenêtre interactive.
- Le mode interactif est confiné à `src/viz/window.rs` (≈ 150 lignes) ; il est entièrement encapsulé derrière la feature `viz-interactive` et ne s'invite pas dans la couche de calcul.
