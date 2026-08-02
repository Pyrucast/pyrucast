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

# Côté Python (pour les tests pytest) : les features passées en ligne de
# commande *remplacent* celles du pyproject, il faut donc redonner
# `extension-module` en plus de `viz`.
maturin develop --features extension-module,viz
```

## Modèle de point de vue

La caméra est décrite par une structure `View`, située sur une sphère orientée autour d'un point cible :

- `yaw` : azimut en degrés (rotation autour de l'axe Z monde) ;
- `pitch` : élévation en degrés (au-dessus du plan XY monde) ;
- `scale` : `1.0` = la bounding-box remplit l'image ; `>1` zoom, `<1` dézoom ;
- `target` : point regardé. `None` ⇒ le centre de la bounding-box de l'objet visualisé ;
- `revolve` : sur une géométrie **axisymétrique** uniquement, balaie la section méridienne pour tracer le corps de révolution (voir [plus bas](#axisymétrie--section-méridienne-ou-corps-de-révolution)). `None` (défaut) ⇒ la section plane.

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

Dans la fenêtre interactive uniquement, le point de vue courant `view=(yaw, pitch, scale)` s'affiche en permanence en haut à droite — le même ordre que le tuple accepté par `view=`, pour recopier tel quel l'angle atteint à la souris/molette dans un appel `plot()` ultérieur.

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

## Nom de figure / de fenêtre (`title`)

Toutes les méthodes `plot(...)` — `Mesh`, `SubMesh`, `NodeField`, `ElementField` — acceptent un argument nommé optionnel `title`, qui sert de **nom de figure** :

- en export fichier (PNG/SVG), il est gravé **centré en bas de l'image**, dans une bande réservée sous le tracé ;
- en fenêtre interactive (`save=None`), il devient le **titre de la fenêtre** (barre de titre de l'OS).

`title=None` (défaut) : aucune légende en bas et titre de fenêtre par défaut (`pyrucast`). Une chaîne vide vaut `None`.

```python
mesh.plot(save="piece.svg", title="poutre encastrée")  # légende centrée en bas du SVG
mesh.plot(save="t.svg", field=t_field, title="température")  # combinable avec field
mesh.plot(title="ma pièce")  # nomme la fenêtre interactive
```

> Pour les **courbes** d'`Evolution` / `SubEvolution`, le `title` existant reste la légende en haut du graphe (voir plus bas) ; il n'est pas repris en bas.

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
sm = pyrucast.Mesh(coords, "TRI3")[0]  # vue du sous-maillage unique
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
mesh.plot(
    save="uy.svg", field=u_field, component="UY", cmap="coolwarm", vmin=-1.0, vmax=1.0
)

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

## Tracé d'une évolution

L'objet [`Evolution` / `SubEvolution`](evolution.md) expose `plot(...)`, qui s'adapte au type de valeur tabulée :

- **évolution de scalaires** → une **courbe X-Y** : l'abscisse est la variable, l'ordonnée la valeur. Un agrégat à plusieurs zones trace **une ligne par zone** avec légende ; chaque échantillon tabulé est marqué d'un point. Les libellés se règlent par `x_label` / `y_label` / `title`. Les arguments de champ (`mesh`, `component`, `cmap`, …) sont sans effet ici.
- **évolution de champs** → le champ est rendu **comme par `mesh.plot(field=...)`**, pour **une valeur tabulée** à la fois. La géométrie suit la même règle que le tracé d'un champ seul : champ par éléments → reconstruit son maillage via le support EF ; champ aux nœuds → **nuage de points** par défaut, ou surface si on passe `mesh=<maillage>`.

```python
import pyrucast as pc

# Courbe scalaire (variable → valeur).
e = pc.Evolution([(0.0, 10.0), (1.0, 20.0), (2.0, 5.0)])
e.plot(save="courbe.svg", x_label="temps", y_label="T", title="évolution de T")

# Évolution d'un champ aux nœuds : un NodeField complet par pas de temps.
ev = pc.Evolution([(0.0, champ_t0), (1.0, champ_t1), (2.0, champ_t2)])
ev.plot(save="frame.png", frame=2)  # une valeur tabulée (défaut : la dernière)
ev.plot(save="frame_surf.png", mesh=maillage)  # rendu surfacique sur un maillage fourni
```

### Slider de valeur tabulée (fenêtre interactive)

En mode interactif (`viz-interactive`, `save=None`), une évolution de champs ouvre la fenêtre avec un **slider** dessiné en bas, qui choisit **quelle valeur tabulée** est affichée (le libellé indique `frame k/n   x=…`) :

- **glisser** le curseur du slider à la souris ;
- **touches `←` / `→`** pour reculer / avancer d'un pas tabulé.

Le **bouton de composante** (clic / `Tab`) et la caméra (rotation, molette, axes via `A`) fonctionnent comme d'habitude ; un clic *sur* le slider est intercepté, ailleurs c'est une rotation. Le slider ne choisit que parmi les valeurs **tabulées** — il n'interpole pas entre elles (pour une valeur intermédiaire, voir `interpolate` sur la page [Évolution](evolution.md)).

> Toutes les sous-évolutions d'un agrégat tracé doivent partager la **même grille d'abscisses** (un index de frame global l'exige) ; sinon `plot` lève une erreur.

## Types d'éléments rendus

**Tous les types d'éléments sont rendus**, chacun converti en une primitive géométrique :

| Type | Primitive | Rendu |
|---|---|---|
| `POI1` | point | un point coloré |
| `SEG2` | segment | une arête |
| `TRI3` | face | triangle plein + contour noir |
| `QUA4` | face | quadrangle plein + contour noir |
| `TET4` | faces triangulaires | la **peau** du volume (facettes de bord) |
| `HEX8` | faces quadrangulaires | la **peau** du volume (facettes de bord) |

`Mesh::plot` parcourt tous ses sous-maillages et dessine chacun selon son type. L'ajout d'un éventuel nouveau type d'élément se fera sans changement d'API, en étendant le `match` de `submesh_primitives` dans `src/viz/mesh_draw.rs`.

### Maillages volumiques pleins

Pour un sous-maillage volumique (`TET4` / `HEX8`), seules les **facettes de bord** sont dessinées : une facette partagée par deux éléments est intérieure au solide, jamais visible, et donc supprimée (une facette est de bord quand elle n'apparaît que dans un seul élément). Cela rend un maillage volumique plein comme une **surface fermée opaque** au lieu d'un enchevêtrement de toutes les facettes internes, et divise à peu près par deux le nombre de primitives.

Les facettes sont tracées **opaques** : combinées au tri en profondeur de l'algorithme du peintre, elles réalisent l'élimination des faces cachées (une facette proche recouvre intégralement celles derrière elle), si bien qu'on ne voit que la peau tournée vers la caméra. La suppression des faces intérieures s'applique aussi bien au tracé géométrique qu'à la coloration par un champ (plate **et** interpolée), de sorte qu'un champ sur un maillage volumique colore correctement sa surface externe.

### Peau opaque ou fil de fer

Pour le tracé **d'un maillage seul** (sans champ), deux styles sont disponibles :

| Style | Rendu |
|---|---|
| `Surface` *(défaut)* | la peau externe opaque ; l'intérieur est masqué |
| `Wireframe` | **toutes** les arêtes en fil de fer (y compris les arêtes intérieures des volumes), sans remplissage — un tracé transparent |

Le fil de fer trace chaque arête distincte (les arêtes partagées par plusieurs cellules ne sont dessinées qu'une fois) dans la `face_color` du sous-maillage, donc les composants d'un `Mesh` restent distinguables. Ce choix **n'a pas de sens pour la coloration par un champ** (un champ peint toujours les faces) : le combiner avec `field` lève une erreur.

Côté Rust, le style est un argument [`MeshStyle`] passé à `plot_styled` :

```rust,ignore
use pyrucast::viz::{MeshStyle, View};
use std::path::Path;

// Peau opaque (équivalent de plot).
mesh.plot_styled(Some(View::iso()), Some(Path::new("solide.svg")), MeshStyle::Surface).unwrap();
// Fil de fer : toutes les arêtes.
mesh.plot_styled(Some(View::iso()), Some(Path::new("fil.svg")), MeshStyle::Wireframe).unwrap();
```

Côté Python, c'est l'argument booléen `wireframe` de `plot` :

```python
mesh.plot(save="solide.svg")  # peau opaque (défaut)
mesh.plot(save="fil.svg", wireframe=True)  # fil de fer

# Sans objet avec un champ : lève ValueError.
# mesh.plot(save="x.svg", field=t_field, wireframe=True)
```

## Axisymétrie : section méridienne ou corps de révolution

Un maillage bâti sur des [coordonnées axisymétriques](coords.md) est le **demi-plan méridien** `(r, z)` d'un corps de révolution : tracé tel quel, il se lit comme une section plane 2-D — ce qui est fidèle à l'objet calculé, mais peu parlant pour montrer la pièce.

L'option `revolve` balaie cette section autour de l'axe `r = 0` et dessine le solide qu'elle décrit. **Rien n'est recalculé** : le balayage a lieu sur les primitives de rendu, juste avant la projection, donc il s'applique de la même façon au maillage seul, au fil de fer, à la coloration par un champ (plate **et** interpolée) et aux évolutions.

| Argument | Défaut | Effet |
|---|---|---|
| `revolve` | `False` | `True` ⇒ trace le corps de révolution au lieu de la section |
| `revolve_angle` | `360.0` | angle balayé en degrés, dans `]0, 360]` |

Un angle **partiel** ouvre la pièce et dessine la section méridienne — et le champ qui la colore — aux deux extrémités du balayage, comme une coupe.

```python
import pyrucast

coords = pyrucast.Coords.axisymmetric()  # (r, z), r ≥ 0
# … maillage de la section, calcul, champ t_field …

mesh.plot(save="section.svg")  # la section plane (défaut)
mesh.plot(save="piece.svg", revolve=True)  # le corps de révolution complet
mesh.plot(save="coupe.svg", revolve=True, revolve_angle=270.0)  # ouvert à 270°
mesh.plot(save="t3d.svg", field=t_field, revolve=True)  # champ sur le corps
```

Côté Rust, c'est le champ `revolve` de la `View`, portant un [`Revolve`] :

```rust,ignore
use pyrucast::viz::{Revolve, View};
use std::path::Path;

let vue = View { revolve: Some(Revolve::full()), ..View::iso() };
mesh.plot(Some(vue), Some(Path::new("piece.svg"))).unwrap();

// Balayage partiel, ou finesse angulaire choisie à la main.
let _ = Revolve::new(270.0).unwrap();              // un secteur par 10°
let _ = Revolve::with_sectors(360.0, 72).unwrap(); // silhouette plus lisse
```

Demander `revolve` sur une géométrie **non** axisymétrique est une erreur : l'abscisse n'y est pas un rayon, le balayage n'aurait aucun sens.

### Ce qui est dessiné

Seul ce qui est **visible** est émis, l'algorithme du peintre faisant le reste :

- une **face** balaie un anneau de matière. Seules les arêtes de **bord** de la section engendrent une surface latérale : une arête partagée par deux cellules reste enfouie dans la matière. C'est le pendant exact de la suppression des facettes intérieures des maillages volumiques ;
- le **contour d'élément** du rendu interpolé suit la même règle, si bien que le quadrillage du maillage reste tracé sur la surface balayée ;
- **segments et points** sont répétés à chaque station angulaire, et les cercles décrits par leurs extrémités sont ajoutés : c'est le fil de fer (resp. le nuage de nœuds) du maillage balayé ;
- une arête posée **sur l'axe** (`r = 0`) ne balaie rien ; une arête qui le touche par une extrémité balaie un cône (triangles au lieu de quadrangles).

### Bascule dans la fenêtre interactive

En mode interactif, sur une géométrie axisymétrique uniquement :

- un **bouton en haut à gauche** indique l'état courant (`2D section` / `3D 360deg`) et bascule au clic ;
- **touche `R`** — même effet, sans toucher à la souris.

La caméra se recentre à chaque bascule : le corps balayé est centré sur l'axe, la section ne l'est pas, sans quoi la pièce sortirait du cadre.

## Export vers ParaView (`export_vtk`)

Pour les maillages industriels — ou simplement pour exploiter les filtres de
**ParaView** — `export_vtk` écrit un fichier **VTK legacy**
(`UNSTRUCTURED_GRID`, ASCII) que ParaView lit nativement. C'est l'opérateur
d'*export* (`src/ops/export`), pendant « écriture » du lecteur `read_gmsh`.

```python
import pyrucast

# Géométrie seule.
pyrucast.export.export_vtk(mesh, "maillage.vtk")

# Géométrie + champ aux nœuds (POINT_DATA).
pyrucast.export.export_vtk(mesh, "solution.vtk", field=temperature)

# Géométrie + champ aux points de Gauss (CELL_DATA) : une valeur par
# cellule = moyenne intra-élément des points de Gauss de la cellule.
pyrucast.export.export_vtk(mesh, "contraintes.vtk", field=stresses)
```

- Chaque sous-maillage est écrit ; les types d'éléments se traduisent un pour
  un (`POI1`→VERTEX, `SEG2`→LINE, `TRI3`→TRIANGLE, `QUA4`→QUAD, `TET4`→TETRA,
  `HEX8`→HEXAHEDRON) et l'ordre local des nœuds coïncide déjà avec celui de
  VTK : la connectivité est copiée telle quelle.
- Une `Coords` 2-D est complétée en 3-D avec `z = 0`.
- Un `NodeField` donne un tableau `SCALARS` par composante aux **points**
  (valeur nodale, `0` là où le champ n'est pas défini) ; un `ElementField`
  donne un tableau par composante aux **cellules**. La valeur par cellule est
  la moyenne des points de Gauss de *cette* cellule (moyenne **intra**-élément
  uniquement — les discontinuités inter-éléments restent visibles). Le champ
  aux éléments doit provenir d'un espace bâti sur **ce** maillage (cellules
  alignées une à une).

Côté Rust : `ops::export::write_vtk_mesh`, `write_vtk_node_field`,
`write_vtk_element_field` (et leurs variantes `vtk_*_string` qui rendent le
texte sans toucher au disque).

### Limites actuelles et évolutions possibles

Cette première version vise la simplicité et la portabilité. Limites assumées,
et les directions pour les lever :

- **VTK legacy ASCII uniquement.** Pas de `.vtu` (XML), pas de variante
  binaire ni de compression : les fichiers sont donc volumineux et l'écriture
  reste en texte. Évolutions : un back-end `.vtu` (recommandé par ParaView,
  extensible), puis un encodage binaire/compressé pour les gros maillages.
- **Composantes en scalaires séparés.** Chaque composante donne un tableau
  `SCALARS` distinct ; pas de regroupement en `VECTORS`/`TENSORS`. Un
  déplacement `(ux, uy, uz)` sort en trois scalaires plutôt qu'en un champ
  vectoriel directement « warpable » dans ParaView. Évolution : détecter/grouper
  les composantes vectorielles et tensorielles.
- **`CELL_DATA` = moyenne des points de Gauss.** Un champ aux éléments est
  réduit à **une** valeur par cellule (moyenne intra-élément). Pas d'export
  des valeurs nodales reconstruites par élément ni des points de Gauss
  individuels. Évolution : écrire les valeurs ajustées par élément (comme la
  viz) en `POINT_DATA` discontinu, ou un VTK à plusieurs points de Gauss.
- **Un seul `Mesh`, un seul champ par fichier.** Pas de séries temporelles
  (`PVD`/`.vtu` multi-pas) ni de plusieurs champs simultanés. Évolutions :
  accepter plusieurs champs en une passe, et une série temporelle pour les
  calculs transitoires (cf. `PASAPAS`).
- **Cellules alignées requises pour `CELL_DATA`.** Le champ aux éléments doit
  provenir d'un espace bâti sur **ce** maillage (correspondance cellule à
  cellule, vérifiée par un simple comptage). Évolution : un appariement
  explicite maillage ↔ espace EF plutôt qu'un ordre implicite.

## Notes techniques

- Le rendu utilise l'algorithme du **peintre** : projection 3D → 2D, tri des triangles par profondeur moyenne (du plus lointain au plus proche), puis dessin des facettes pleines **opaques** suivies des arêtes noires en superposition. L'opacité assure l'élimination des faces cachées (les facettes proches recouvrent les lointaines) ; c'est ce qui fait qu'un solide 3D se lit comme un solide et non comme une coque transparente. Coût : `O(n log n)` à chaque rafraîchissement, raisonnable jusqu'à quelques milliers de cellules. Pour des maillages plus lourds ou un post-traitement avancé, exporter vers ParaView avec `export_vtk` (voir ci-dessus).
- Limite connue de l'algorithme du peintre : pour un solide **fortement non convexe**, le tri par profondeur moyenne peut mal ordonner deux facettes qui se chevauchent en profondeur. C'est inhérent à la méthode ; un *z-buffer* par pixel le corrigerait, au prix d'un rendu non vectoriel.
- Le balayage axisymétrique (`revolve`) multiplie le nombre de primitives par le nombre de secteurs, mais **seulement sur le bord** de la section (les arêtes intérieures ne balaient rien) : le coût reste proportionnel au périmètre, pas à la surface. Une section partielle ajoute en plus une copie de la section à chaque extrémité.
- L'export reste **portable Linux ↔ Windows** : tout le rendu se fait en CPU, sans pilote GPU. Le binaire `viz-interactive` nécessite en revanche un serveur d'affichage (X11, Wayland ou Windows) à l'exécution — ce qui est attendu pour une fenêtre interactive.
- Le mode interactif est confiné à `src/viz/window.rs` ; il est entièrement encapsulé derrière la feature `viz-interactive` et ne s'invite pas dans la couche de calcul.
