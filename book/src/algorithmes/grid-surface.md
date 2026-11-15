# Cœur en grille

## Cœur en grille, bande frontale : `grid_surface`

{{#pyalgo mesh.grid_surface}}

`grid_surface(contour, element_type, size=None, band=0, prefer_quads=False,
relax="free")` prend la même entrée que `pave_surface`, rend la même chose, et
tient la même promesse — **le contour est intouchable**. Seul l'intérieur est
obtenu autrement.

### Pourquoi

Les deux familles de mailleur quadrangulaire échouent aux endroits **opposés**.

Un **front** est parfait là où il part : sa première rangée épouse exactement
le contour. Il est douteux là où deux de ses rangées se rencontrent, parce
qu'il lui faut y réconcilier deux discrétisations qui n'ont aucune raison de
s'accorder. C'est cette ligne-là qui porte les défauts de valence, les
triangles résiduels et les mailles aplaties.

Une **grille** est le miroir exact : chaque maille qu'elle pose est un
rectangle par construction, et toute sa difficulté est au bord.

Prendre l'intérieur de la grille et le bord du front ne laisse donc ni l'une ni
l'autre faiblesse.

### Méthode

1. **Les lignes de la grille viennent du contour**, pas de la boîte
   englobante. Toute arête axiale assez longue pour être une caractéristique
   fixe une ligne à sa coordonnée, et les intervalles entre lignes
   consécutives sont subdivisés uniformément à peu près à `size`. Un contour
   sans aucune arête axiale — un cercle — ne fixe rien et on retombe
   proprement sur la grille uniforme.
2. **Les mailles entièrement dans la matière sont gardées.** Une arête de bord
   posée exactement sur une ligne de grille ne coupe *aucune* des deux mailles
   qu'elle sépare : elle passe entre elles.
3. **Un nœud de grille qui tombe sur un nœud du contour *est* ce nœud** — le
   même sommet, pas une copie. Le cœur rejoint donc le bord au lieu de
   s'arrêter à un cheveu de lui.
4. **Le cœur recule seulement là où il ne rejoint pas le contour.** Une face
   du cœur dont les deux extrémités sont des nœuds du contour et qui est un
   segment du contour est finie ; ailleurs le cœur s'arrête *près* du bord, à
   une distance quelconque entre zéro et une maille, et la maille derrière
   cette face est retirée pour laisser au front une maille pleine où
   travailler. `band` en retire davantage.
5. **La bande à paver est le contour et le bord du cœur moins les arêtes
   communes** : un segment parcouru une fois dans chaque sens ne borne rien.
   Sur un domaine rectilinéaire posé sur la grille, il ne reste rien du tout,
   aucun front ne tourne, et le maillage *est* la grille.
6. **Une boucle restante entièrement issue du cœur est gelée.** Elle reste
   vivante — le front la voit, s'en écarte, se coud dessus — mais ne pose
   aucune rangée. Deux fronts vivants se rencontrent où ça tombe ; un seul
   front **atterrit**, sur une interface qu'on a choisie.

### L'orientation est détectée sur le contour

Rien dans le contrat n'attache la grille aux axes du repère : son orientation
est un choix purement interne. Elle est donc prise **sur le contour** — l'angle
qui rend axiale la plus grande longueur de contour, cherché sur un quart de
tour, ce que la symétrie d'ordre 4 d'une grille laisse de distinct.

Sans cela la méthode ne serait qu'un tour de passe-passe ne marchant que sur
les formes qu'on a dessinées d'équerre. Rectangle 0,6 × 0,3, taille 0,02 :

| angle | avant la détection | après |
|---|---|---|
| 0° | 450 mailles, 0 triangle, qualité 1,000 | identique |
| 5° | 496 (28 tri), 64 % de mailles parfaites | **450, 0 tri, 100 %** |
| 15° | 470 (20 tri), 62 % | **450, 0 tri, 100 %** |
| 30° | 454 (20 tri), 61 % | **450, 0 tri, 100 %** |
| 45° | 500 (30 tri), 55 % | **450, 0 tri, 100 %** |

À 30° la pire maille était moins bonne qu'avec `pave_surface` seul. Le profil
crénelé tourné de 23,7°
retrouve lui aussi ses 4 032 mailles parfaites.

La boîte englobante s'en trouve **resserrée**, pas agrandie : elle est calculée
sur les points déjà tournés. À 30° elle passe de 0,670 × 0,560 — soit 2,08 fois
l'aire de la pièce, autant de mailles classifiées pour rien — à 0,6 × 0,3.

Deux garde-fous :

- un contour **sans direction dominante** — un cercle, dont les angles d'arête
  sont répartis uniformément — garde les axes : tourner échangerait une
  orientation arbitraire contre une autre, au prix de la reproductibilité ;
- **les axes gagnent les égalités.** Une forme déjà d'équerre est le cas que
  cette détection ne doit surtout pas casser, et il serait mauvais de le perdre
  pour un gain de la largeur d'un arrondi ailleurs. Un chanfrein à 30° sur une
  pièce par ailleurs droite ne fait donc pas tourner toute la grille.

Reste le cas des **directions concurrentes** : une pièce mêlant des bords à 0°
et à 30° ne peut satisfaire que l'une des deux familles. C'est la plus longue
qui est servie.

### Le contour doit être discrétisé pour une grille

C'est la seule chose demandée à l'appelant, et c'est une vraie contrainte. Une
grille ne peut rejoindre qu'un contour dont les nœuds tombent sur ses lignes,
et ses lignes sont dictées par la forme elle-même.

Soit un profil de 0,6 de large fait de neuf créneaux. Les créneaux posent une
ligne tous les 0,0667 ; des mailles de 0,00375 donnent donc 18 colonnes par
créneau, à 0,0037037. Une base discrétisée d'un seul tenant en 160 segments de
0,00375 les manque **toutes**, de 1,2 % — assez pour qu'aucun nœud ne soit
partagé et que tout le bord retombe sur le front. Couper cette base sous chaque
créneau, pour que chaque morceau prenne ses 18 segments, ne coûte rien et
partage tous les nœuds.

La règle est donc : **couper chaque côté aux angles de la forme, et laisser
chaque morceau prendre un nombre entier de mailles.** Rien ne le vérifie, parce
qu'un contour qui ne le fait pas n'est pas une erreur — il obtient simplement
plus de bande et moins de grille.

### Exemple Python

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-grid_surface}}
```

### Qualité

Le profil crénelé de 0,6 × 0,3 à sept angles rentrants, taille visée 0,00375 :

| | `pave_surface` | `grid_surface` |
|---|---|---|
| mailles | 6 428 QUA4 + 26 TRI3 | **4 032 QUA4, 0 TRI3** |
| qualité — min | 0,229 | **1,000** |
| qualité — médiane | 0,854 | **1,000** |
| mailles sous 0,7 | 3,13 % | **0** |
| aire couverte | exacte | exacte |

Un tiers de mailles en moins pour une qualité parfaite. Sur un rectangle
0,6 × 0,3 à 0,02 : 450 mailles de qualité 1, contre 650 et 6 triangles.

### Pièges

- **`band` vaut 0 et c'est la bonne valeur.** Le cœur recule déjà d'une maille
  partout où il ne rejoint pas le contour. Ne l'augmentez que pour éloigner
  volontairement l'interface du bord.
- **`relax` gouverne la bande, pas le cœur** — la grille est posée droite quoi
  qu'il arrive. Voir [la relaxation du front](pave-surface.md#relaxation-du-front--relax) :
  sur une bande épaisse et rectilinéaire, `"along"` la garde structurée.
- **Pas de gradation, et c'est un choix.** La grille est uniforme par
  intervalle entre lignes. Un quadtree la graduerait — règle 2:1 et gabarits
  de transition — mais le résultat ne vaut pas ce qu'il coûte, et la raison
  est géométrique, pas algorithmique : passer de \\( n \\) nœuds à
  \\( n/2 \\) le long d'une interface **en gardant tous les angles droits est
  impossible**. La maille qui absorbe la réduction a trois nœuds d'un côté et
  deux de l'autre — ce n'est plus un rectangle, par construction. Tout
  raffinement local conforme paie donc en angles à 45°, et un quadtree en
  paie beaucoup : la gradation par distance au bord fait des couronnes
  concentriques, chaque feuille de couronne a **un** côté raffiné, un bord à
  cinq arêtes n'admet aucun découpage en quadrangles (dans tout
  quadrillage \\( 4Q = 2E_\text{int} + E_\text{bord} \\), donc
  \\( E_\text{bord} \\) est pair), et il faut un triangle par feuille. Mesuré
  sur un rectangle : 60 % de mailles en moins, mais un dixième de triangles et
  un jacobien minimal qui tombe de 1,000 à 0,70.

  La seule gradation compatible avec des rectangles parfaits est celle qui ne
  change **aucun** compte de nœuds : espacer les lignes de la grille en
  progression géométrique. Elle est par axe — une colonne fine traverse toute
  la hauteur — donc pas localisable, mais elle garde le jacobien à 1. Ce n'est
  pas fait non plus.

### Le contour, nœud par nœud

La garantie est plus forte que « le bord est respecté » : **tous les nœuds du
contour reviennent, à leur position, avec leur identité**, et le maillage n'a
aucune autre arête de bord. Un mailleur en grille ne l'hérite pas — il pose ses
propres nœuds — il ne la tient que parce qu'un nœud de grille tombant sur un
nœud du contour *est* ce nœud.

Deux drapeaux distincts le portent, et les confondre coûtait cher :

- **immobile** — le nœud ne doit pas bouger au lissage : ceux du contour, et
  ceux du cœur, qui sont d'équerre par construction et n'ont rien à y gagner ;
- **inaliénable** — rien ne doit l'abandonner : ceux du contour, et eux seuls.
  Un nœud du cœur est à nous ; la soudure peut le céder, et c'est ce qui lui
  permet de refermer une lèvre entre cœur et contour. Tant que les deux
  drapeaux n'en faisaient qu'un, ces lèvres restaient ouvertes : sur un cercle,
  **88 arêtes de bord** n'appartenaient pas au contour, soit 22 trous.

### Dégradation

Une région trop mince pour tenir une seule maille de cœur n'en reçoit aucune,
et le front la pave seul, exactement comme `pave_surface`. Un contour hors
grille reçoit une bande large et le même traitement. La dégradation est
**continue** : au pire la qualité du paveur frontal.

Continue, mais bornée par le bas par la **validité**. Si le front s'emmêle au
point de laisser une maille retournée ou plate, l'opérateur **refuse** au lieu
de rendre le maillage : un jacobien négatif n'est pas une maille médiocre mais
une maille fausse, qu'aucun code éléments finis n'intègre. Mieux vaut
l'apprendre ici qu'à l'assemblage.

Sur un cercle — le cas dégradé par excellence, aucune arête axiale — il ne
reste **aucune fissure** : toutes les arêtes de bord sont celles du contour.
Il manque 0,18 % d'aire, les lèvres que la soudure referme. Ce n'était pas
acquis : tant que la retraite du front pouvait poser des rangées à 1,5 % de la
taille demandée, ces rangées laissaient 16 arêtes orphelines, soit 4 trous
d'une maille.

### La phase de la grille est déjà la bonne

Une fois l'orientation trouvée, on pourrait vouloir **translater** la grille
selon ses axes pour améliorer la bande. Deux raisons de ne pas le faire, la
seconde mesurée.

D'abord, la phase n'est pas libre dans les cas visés : le calage sur les
arêtes du contour *est* un mécanisme de phase. Une ligne épinglée par une
arête verticale passe exactement par elle, et `fill` subdivise chaque
intervalle séparément, si bien que chaque ligne de forme est atteinte
exactement. Il n'y a rien à gagner là où l'opérateur brille.

La phase n'est donc libre que lorsque rien n'est épinglé — une courbe. Un
balayage de huit phases sur un cercle a été mesuré : **la phase actuelle gagne
sur tous les critères** — le moins de mailles, le moins de triangles, et la
seule à n'ouvrir aucune fissure.

Ce n'est pas un hasard : ancrer sur le coin de la boîte englobante fait passer
les lignes de grille exactement par les points extrêmes du contour, qui sont
ses points de tangence. Une phase quelconque les manque, et chaque ligne
manquée se paie en mailles de raccord.

## Lignes par nœud, rangées pliées : `grid_surface2`

{{#pyalgo mesh.grid_surface2}}

`grid_surface2(contour, element_type, size=None, band=0, prefer_quads=False,
relax="free")` prend les mêmes contours, rend le même maillage, respecte le
même contour intouchable que `grid_surface`. **Il ne le remplace pas** : la façon de poser la grille
diffère, et aucune des deux ne gagne partout.

### La différence

`grid_surface` pose une ligne sur la coordonnée où repose chaque côté aligné,
puis découpe l'espace entre deux lignes d'après le côté qui l'enjambe de bout en
bout. Toutes les lignes sont droites et toute maille du cœur est un rectangle.

`grid_surface2` donne à **chaque nœud du contour la ligne qui le traverse**,
puis :

- coupe en mailles entières tout intervalle valant entre deux et trois fois le
  pas moyen que porte le contour, et **abandonne au-delà** : un vide que le
  contour ne borde pas revient au front d'un seul tenant plutôt qu'en rangées
  qui n'existent que pour être érodées ;
- **effondre arête par arête** toute bande plus mince que la moitié de ce pas —
  chaque arête se soude sur le nœud de contour d'une de ses extrémités, ou sur
  son milieu quand aucune n'en est un ;
- va chercher le reste : un nœud de grille à moins d'un quart de maille d'un
  nœud du contour se déplace dessus ;
- et juge chaque maille sur la forme qu'elle a finalement, pas sur les lignes
  dont elle vient.

Une rangée est donc une **polyligne**, pas une ligne. C'est tout l'intérêt :
une même rangée peut rejoindre deux parois qui se font face à deux hauteurs
différentes, ce qu'aucune droite ne sait faire. Une paroi découpée en dix peut
ainsi faire face à une paroi découpée en onze — `grid_surface` doit en choisir
une et envoie l'autre à la bande.

### Lequel prendre

Mesuré sur les mêmes formes, à la même taille visée, pire maille en *mean
ratio* :

| forme | `grid_surface` | `grid_surface2` |
|---|---|---|
| rectangle, et toute forme pile sur la grille | **0,999** | **0,999** |
| plaque à marche hors grille | 0,405 | **0,963** |
| L à cotes quelconques | 0,437 | **0,979** |
| L dont les côtés découpent 5+6 contre 4+7 | 0,421 | **0,963** |
| L dont les côtés diffèrent d'un nœud | 0,351 | **0,606** |
| profil crénelé, base coupée sous chaque barre | 0,382 | **0,916** |
| profil crénelé, base d'un seul tenant | 0,323 | **0,651** |
| maison à toit à deux pentes | 0,420 | **0,548**, 1 triangle contre 11 |
| carré à un angle arrondi | 0,244 | **0,400** |
| cercle R = 1 | **0,288**, p5 **0,796** | **0,005 — à ne pas utiliser** |

La comparaison des quatre mailleurs surfaciques, figures à l'appui, est sur la
page [Mailler une géométrie](../mailler.md).

Donc : **`grid_surface2` pour une forme rectilinéaire**, d'autant plus si ses
côtés n'ont pas été coupés aux angles qui leur font face ; **`grid_surface`
pour tout ce qui est courbe**, où suivre les nœuds du contour revient à suivre
le hasard de l'endroit où ses sommets sont tombés.

### Exemple Python

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-grid_surface2}}
```
