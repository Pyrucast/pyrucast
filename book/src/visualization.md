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
use pyrucast::containers::mesh::Coords;
use pyrucast::mesh::element_type::ElementType;
use pyrucast::mesh::SubMesh;
use pyrucast::mesh::node::Node;
use pyrucast::store::insert;
use pyrucast::viz::View;
use std::path::Path;

let coords = insert(Coords::new(3).unwrap());
let a = Node::create_in(coords.clone(), &[0.0, 0.0, 0.0]).unwrap();
let b = Node::create_in(coords.clone(), &[1.0, 0.0, 0.0]).unwrap();
let c = Node::create_in(coords.clone(), &[0.0, 1.0, 0.0]).unwrap();
let mut sm = SubMesh::new(coords, ElementType::TRI3);
sm.add_cell(&[a.id(), b.id(), c.id()]).unwrap();

// Export vectoriel.
sm.plot(Some(View::iso()), Some(Path::new("triangle.svg"))).unwrap();
// Fenêtre interactive (feature `viz-interactive`).
// sm.plot(None, None).unwrap();
```

Côté Python, l'API miroir prend des tuples :

```python
import pyrucast

coords = pyrucast.Coords(3)
a = coords.add_node([0.0, 0.0, 0.0])
b = coords.add_node([1.0, 0.0, 0.0])
c = coords.add_node([0.0, 1.0, 0.0])

mesh = pyrucast.Mesh(coords, "TRI3")
mesh.unit().add_cell([a, b, c])

# (yaw, pitch, scale) ; save=None ouvre la fenêtre interactive.
mesh.plot(view=(45.0, 35.264, 1.0), save="triangle.svg")
```

## Couleur de face par `SubMesh`

Chaque `SubMesh` porte une propriété `face_color` (type `RgbColor`, format `(r, g, b)` sur 8 bits) utilisée par la couche viz pour remplir les facettes. Cette donnée n'a **aucun effet sur les calculs** ; elle est simplement persistée avec le maillage et consommée par `plot`. Couleur par défaut : un bleu clair (`180, 200, 230`).

Côté Rust :

```rust,ignore
use pyrucast::mesh::color::RgbColor;
use pyrucast::mesh::element_type::ElementType;
use pyrucast::mesh::SubMesh;

let mut sm = SubMesh::new(coords, ElementType::TRI3);
sm.set_face_color(RgbColor::new(220, 60, 60));
assert_eq!(sm.face_color(), RgbColor::new(220, 60, 60));
```

Côté Python :

```python
sm = pyrucast.Mesh(coords, "TRI3")[0]   # vue du sous-maillage unique
sm.face_color = (220, 60, 60)
assert sm.face_color == (220, 60, 60)
```

Quand on appelle `Mesh::plot`, chaque sous-maillage est rendu avec **sa propre** `face_color`, ce qui permet de distinguer visuellement des composants regroupés dans un même maillage (par exemple : peau / cœur / interfaces).

## Coloration par un champ — `NodeField` ou `ElementField`

`plot` accepte un argument optionnel `field` — un [`NodeField`](node-field.md) **ou** un [`ElementField`](element-field.md), interchangeables — qui **remplace la couleur uniforme** par une couleur tirée d'une *colormap* appliquée aux valeurs du champ.

Le rendu raisonne **par élément** : pour dessiner un élément, il lui faut des valeurs à ses nœuds, propres à cet élément.

- **`NodeField`** : les valeurs nodales sont lues directement (champ continu par construction ; un nœud absent du support prend la moyenne des nœuds présents).
- **`ElementField`** : les valeurs vivent aux points de Gauss. Les valeurs nodales du tracé viennent d'un **moindre carré local à l'élément** (fit de l'interpolant Lagrange aux valeurs de Gauss de *cet* élément). Aucune moyenne entre éléments voisins : **les discontinuités inter-éléments — flux, contraintes — restent visibles**, c'est une information physique. Avec un seul point de Gauss, le fit dégénère en couleur constante par élément.

Affichage commun aux deux types :

- **Composante affichée** : la première composante du champ par défaut ; on en choisit une autre via `component="<nom>"`.
- **Échelle** : linéaire entre le minimum et le maximum **observés** sur le maillage rendu, sauf si on fixe les bornes (voir [Bornes](#bornes-de-léchelle-vmin--vmax)).
- **Colorbar** : une barre verticale graduée est dessinée sur le bord droit de l'image (bas = borne basse, haut = borne haute), avec le même dégradé que les cellules.
- **Bandeau** : en haut de l'image, le nom de la composante affichée et l'intervalle `[min, max]`.

### Rendu interpolé (`smooth`)

Par défaut (`smooth=4`), la couleur **suit les fonctions de forme à l'intérieur de chaque élément** : chaque maille est sous-découpée en sous-triangles (TRI3 → n², QUA4 → 2n²) dont la géométrie et la valeur sont évaluées par `N_i(ξ)` — y compris le gauchissement bilinéaire des QUA4/HEX8. Le filaire noir n'est tracé que sur les arêtes d'origine des éléments. `smooth=0` revient à une couleur plate par cellule (moyenne des valeurs nodales) ; monter `smooth` lisse davantage au prix de `n²` polygones par élément.

Le sous-découpage est purement graphique et interne à chaque élément : les sous-sommets d'une arête partagée sont évalués séparément de chaque côté, donc les discontinuités d'un `ElementField` traversent le rendu interpolé sans être gommées.

### Tracé d'un champ seul

- `element_field.plot(...)` fonctionne sans maillage : chaque zone retrouve son sous-maillage via son sous-espace EF (partagé, pas copié).
- `node_field.plot(...)` trace un **nuage de points colorés** : son support POI1 ne porte pas de connectivité, aucune surface ne peut être inférée — pour des surfaces, passer par `mesh.plot(field=...)` avec le maillage d'origine.

### Échelles de couleur (`cmap`)

Cinq colormaps sont disponibles, sélectionnées par leur nom (insensible à la casse). Un nom inconnu lève une erreur listant les noms acceptés.

| `cmap` | Dégradé | Usage |
|---|---|---|
| `"viridis"` *(défaut)* | violet → bleu → vert → jaune | usage général ; perceptuellement uniforme, lisible en niveaux de gris et pour les daltoniens |
| `"coolwarm"` | bleu → blanc → rouge | données signées centrées sur 0 (le blanc marque le milieu de l'échelle) |
| `"hot"` | noir → rouge → jaune → blanc | rendu thermique |
| `"gray"` | noir → blanc | impression N&B, superposition |
| `"jet"` | bleu → vert → rouge | ancien défaut, conservé |

### Bornes de l'échelle (`vmin` / `vmax`)

Par défaut l'échelle couvre le min/max des valeurs **par cellule**. On peut fixer l'une ou l'autre borne (ou les deux) — celle laissée à `None` continue de suivre les données. Utile pour **comparer plusieurs figures** sur une échelle commune, ou pour centrer une colormap divergente (`vmin = -vmax` avec `"coolwarm"`).

Côté Rust, l'argument `scale` (`ColorScale`) regroupe colormap et bornes :

```rust,ignore
use pyrucast::containers::node_field::SubNodeField;
use pyrucast::viz::{Colormap, ColorScale};

// Champ déplacement à 2 composantes "UX" / "UY" sur un POI1 (la couche
// viz Rust consomme la zone ; côté Python, plot prend le NodeField).
let mut u = SubNodeField::from_poi1(&poi1_h, vec!["UX".into(), "UY".into()]).unwrap();
// ... remplissage ...

// Échelle auto, viridis, première composante, rendu interpolé niveau 4.
use pyrucast::viz::FieldArg;
mesh.plot_with_field(None, Some(std::path::Path::new("ux.svg")),
    FieldArg::Node(&u), None, ColorScale::default(), 4).unwrap();

// Composante "UY", colormap coolwarm, bornes fixées à [-1, 1], plat.
let scale = ColorScale { cmap: Colormap::CoolWarm, vmin: Some(-1.0), vmax: Some(1.0) };
mesh.plot_with_field(None, Some(std::path::Path::new("uy.svg")),
    FieldArg::Node(&u), Some("UY"), scale, 0).unwrap();

// Champ aux points de Gauss : même appel, FieldArg::Element.
mesh.plot_with_field(None, Some(std::path::Path::new("flux.svg")),
    FieldArg::Element(&flux), None, ColorScale::default(), 4).unwrap();
```

Côté Python, `cmap`, `vmin` et `vmax` sont des arguments nommés de `plot` :

```python
# Composante par défaut, viridis, échelle auto.
mesh.plot(save="t.svg", field=t_field)

# Composante "UY", colormap "coolwarm", bornes fixées.
mesh.plot(save="uy.svg", field=u_field, component="UY", cmap="coolwarm", vmin=-1.0, vmax=1.0)

# Plafond seul fixé : le plancher suit le minimum des données.
mesh.plot(save="t.svg", field=t_field, vmax=100.0)

# Champ aux points de Gauss : strictement le même appel.
mesh.plot(save="flux.svg", field=flux_field)

# Couleur plate par cellule (comportement historique).
mesh.plot(save="t_flat.svg", field=t_field, smooth=0)

# Champ seul : l'ElementField reconstruit son maillage ; le NodeField
# trace un nuage de points (pas de connectivité dans son support).
flux_field.plot(save="flux_alone.svg")
t_field.plot(save="t_points.svg")
```

### Bouton de sélection dans la fenêtre interactive

En mode interactif (`viz-interactive`), un **bouton cliquable** apparaît au sommet de la fenêtre, affichant la composante actuelle et son intervalle. Deux manières équivalentes d'en changer :

- **Clic** sur le bouton — cycle dans l'ordre des composantes du champ ;
- **Touche `Tab`** — même effet, sans toucher à la souris.

La caméra (rotation à la souris, molette, axes affichés via `A`) continue de fonctionner exactement comme en plot classique ; seul un clic *sur* le bouton est intercepté, les clics ailleurs lancent une rotation comme d'habitude.

## Types d'éléments rendus

**Tous les types d'éléments sont rendus**, chacun converti en une primitive géométrique :

| Type | Primitive | Rendu |
|---|---|---|
| `POI1` | point | un point coloré |
| `SEG2` | segment | une arête |
| `TRI3` | face | triangle plein + contour noir |
| `QUA4` | face | quadrangle plein + contour noir |
| `TET4` | 4 faces | les 4 facettes triangulaires du volume |
| `HEX8` | 6 faces | les 6 facettes quadrangulaires du volume |

`Mesh::plot` parcourt tous ses sous-maillages et dessine chacun selon son type. L'ajout d'un éventuel nouveau type d'élément se fera sans changement d'API, en étendant le `match` de `submesh_primitives` dans `src/viz/mesh_draw.rs`.

## Notes techniques

- Le rendu utilise l'algorithme du **peintre** : projection 3D → 2D, tri des triangles par profondeur moyenne (du plus lointain au plus proche), puis dessin des facettes pleines suivies des arêtes noires en superposition. Coût : `O(n log n)` à chaque rafraîchissement, raisonnable jusqu'à quelques milliers de cellules. Pour des maillages industriels, prévoir un export `.vtu` vers ParaView.
- L'export reste **portable Linux ↔ Windows** : tout le rendu se fait en CPU, sans pilote GPU. Le binaire `viz-interactive` nécessite en revanche un serveur d'affichage (X11, Wayland ou Windows) à l'exécution — ce qui est attendu pour une fenêtre interactive.
- Le mode interactif est confiné à `src/viz/window.rs` ; il est entièrement encapsulé derrière la feature `viz-interactive` et ne s'invite pas dans la couche de calcul.
