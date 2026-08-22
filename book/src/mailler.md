# Mailler une géométrie

Cette page est un **guide de choix**, pas une référence : elle compare les
mailleurs entre eux et dit lequel prendre. La description de chaque opérateur,
ses arguments et ses pièges vivent dans
[Opérateurs → Maillage](operateurs/maillage.md).

La partie 3D viendra plus tard ; seul le 2D est traité ici.

## Mailler en 2D

Quatre opérateurs remplissent l'intérieur d'un contour fermé. Ils prennent tous
la même chose — un ou plusieurs [contours orientés](operateurs/maillage.md) en
`SEG2` et une taille de maille visée — et rendent tous un maillage dont le bord
**est** le contour, nœuds compris.

| opérateur | méthode | rend |
|---|---|---|
| `triangulate_surface` | Delaunay contraint + raffinement de Ruppert | des `TRI3` |
| `pave_surface` | front avançant, en rangées depuis le bord | des `QUA4`, quelques `TRI3` |
| `grid_surface` | cœur en grille + bande frontale au bord | des `QUA4`, quelques `TRI3` |
| `grid_surface2` | idem, lignes prises une par nœud du contour | des `QUA4`, quelques `TRI3` |

### Comment lire les figures

Chaque figure montre le **même contour** maillé quatre fois. Il est construit
une seule fois, puis dupliqué par `translate` : aucun mailleur ne bénéficie
d'une discrétisation de bord différente des autres. La disposition ne change
jamais — en haut `triangulate_surface` et `pave_surface`, en bas `grid_surface`
et `grid_surface2`. Le contour est en bleu et ses nœuds en rouge.

Les figures et les chiffres de cette page sortent d'un seul script,
[`examples/comparer_mailleurs_2d.py`](#reproduire-les-figures).

La qualité citée est la **mean ratio du pire coin** : 1 pour un coin droit à
côtés égaux, 0 pour un coin plat, négatif pour une maille retournée. C'est la
même mesure pour les triangles et pour les quadrangles, ce qui est la seule
façon de comparer les quatre sur un pied d'égalité. On donne aussi le
**5ᵉ centile**, qui dit ce que valent les mailles médiocres et non la seule
plus mauvaise.

En pratique cette mesure ne descend jamais à zéro dans les tableaux qui suivent,
et ce n'est pas une chance : **les quatre mailleurs refusent de rendre un
maillage portant une maille retournée ou plate**. Une telle maille a un jacobien
négatif ou nul, qu'aucun code éléments finis n'intègre — ce n'est pas un
maillage médiocre mais un maillage faux, et mieux vaut une erreur qui situe la
maille fautive. Un mailleur qui échoue là où un autre passe est donc un résultat
en soi, à lire dans les tableaux comme tel.

### Une forme rectilinéaire posée sur la grille

C'est le cas le plus favorable aux mailleurs en grille, et la différence est
franche.

![Rectangle](img/mailleurs-2d-rectangle.png)

![Plaque dont la marche tombe sur la grille](img/mailleurs-2d-plaque-sur-grille.png)

| forme | | triangulate | pave | grid | grid2 |
|---|---|---:|---:|---:|---:|
| rectangle 1 × 0,63 | mailles | 286 | 98 | **60** | **60** |
| | pire | 0,425 | 0,465 | **0,999** | **0,999** |
| plaque, marche sur la grille | mailles | 394 | 138 | **80** | **80** |
| | pire | 0,489 | 0,290 | **1,000** | **1,000** |

Les deux mailleurs en grille rendent le maillage qu'on dessinerait à la main :
toutes les mailles sont des rectangles et il n'y a pas un seul triangle. Le
paveur, lui, rend une pelure d'oignon avec quatre coutures diagonales — sa
faiblesse est là où deux de ses rangées se rencontrent, et cette ligne-là existe
même sur un rectangle.

### Une forme rectilinéaire qui ne tombe pas sur la grille

Dès que les cotes ne sont plus des multiples de la taille visée, `grid_surface`
et `grid_surface2` se séparent nettement.

![Plaque dont la marche est hors grille](img/mailleurs-2d-plaque-hors-grille.png)

![L à cotes quelconques](img/mailleurs-2d-l-quelconque.png)

![L étiré à 1,02](img/mailleurs-2d-l-102.png)

![L étiré à 1,10](img/mailleurs-2d-l-110.png)

| forme | | triangulate | pave | grid | grid2 |
|---|---|---:|---:|---:|---:|
| plaque à marche (0,53 ; 0,61) | mailles | 366 | 136 | 86 | **80** |
| | pire | 0,481 | 0,187 | 0,405 | **0,963** |
| L à cotes quelconques | mailles | 308 | 117 | 76 | **70** |
| | pire | 0,459 | 0,446 | 0,437 | **0,979** |
| L étiré à 1,02 | mailles | 319 | 117 | 80 | **74** |
| | pire | 0,477 | 0,551 | 0,351 | **0,606** |
| L étiré à 1,10 | mailles | 336 | 122 | 80 | **74** |
| | pire | 0,474 | 0,446 | 0,421 | **0,963** |

`grid_surface` pose une ligne sur la coordonnée où repose chaque côté aligné,
puis découpe entre deux lignes d'après le côté qui les enjambe : toutes ses
lignes sont droites. `grid_surface2` donne à chaque nœud du contour la ligne qui
le traverse et laisse ses rangées **plier** pour aller chercher le contour — une
même rangée peut alors rejoindre deux parois qui se font face à deux hauteurs
différentes, ce qu'aucune droite ne sait faire.

Le L étiré à 1,02 est le seul de la série que `grid_surface2` ne règle pas :
les deux côtés y posent un nombre de nœuds **différent** sur la même portée,
dix contre onze, et aucune disposition de lignes n'invente la rangée manquante.
Le remède est dans le contour, pas dans le mailleur — voir
[la règle de discrétisation](operateurs/maillage.md).

### Un profil à sept angles rentrants

Le profil crénelé est décliné en deux versions dont **seule la base change** :
coupée sous chaque barre, ou d'un seul tenant. C'est le meilleur révélateur de
la sensibilité d'un mailleur à la discrétisation du contour.

![Créneau, base coupée sous chaque barre](img/mailleurs-2d-creneau-base-coupee.png)

![Créneau, base d'un seul tenant](img/mailleurs-2d-creneau-base-entiere.png)

| forme | | triangulate | pave | grid | grid2 |
|---|---|---:|---:|---:|---:|
| base coupée | mailles | 2 050 | 860 | 474 | **456** |
| | pire | 0,412 | 0,366 | 0,382 | **0,916** |
| base d'un seul tenant | mailles | 2 089 | 863 | 486 | **456** |
| | pire | 0,419 | 0,337 | 0,323 | **0,651** |

`grid_surface2` rend **exactement le même nombre de mailles dans les deux cas**.
Les trois autres paient la base d'un seul tenant. C'est la propriété la plus
utile de ce mailleur : il pardonne une discrétisation de contour que les autres
font payer.

### Une forme oblique ou courbe

Ici le classement s'inverse, et c'est le seul endroit où il le fait.

![Maison](img/mailleurs-2d-maison.png)

![Carré à un angle arrondi](img/mailleurs-2d-carre-arrondi.png)

![Cercle](img/mailleurs-2d-cercle.png)

| forme | | triangulate | pave | grid | grid2 |
|---|---|---:|---:|---:|---:|
| maison | mailles | 1 723 | 620 | 470 | **453** |
| | pire | 0,484 | 0,491 | 0,420 | **0,548** |
| carré arrondi | mailles | 1 866 | 678 | 441 | **415** |
| | pire | **0,474** | 0,360 | 0,244 | 0,400 |
| cercle R = 1 | mailles | 6 114 | 2 339 | **1 260** | 2 044 |
| | pire | **0,424** | 0,031 | **0,288** | 0,005 |

Une grille ne peut pas suivre une oblique : elle la découpe en escalier, et tout
ce qui s'en approche est rendu au paveur frontal. Sur la maison, où l'oblique
ne fait que le toit, ce qui reste de rectilinéaire suffit encore à
`grid_surface2` ; sur le cercle, qui n'a plus rien de droit,
`triangulate_surface` est le meilleur des quatre.

**Sur une forme sans direction dominante, `grid_surface2` n'est pas un
candidat.** Il prend l'écartement de ses lignes sur le contour, et un cercle
n'en dicte aucun : rien ne pose de ligne entre `y = 0,24` et `y = 1`, le vide y
vaut quinze fois l'écartement moyen, et le cœur s'effondre — 0,005, une maille
au bord de la dégénérescence. `grid_surface`, lui, rend 0,288 avec un 5ᵉ
centile de 0,796, de loin le meilleur des quatre. **Sur une courbe, prenez
`grid_surface`, ou `triangulate_surface` si la qualité du pire élément
commande.**

Le cercle mérite un mot de plus. `pave_surface` y tombe à **0,031** — ses
rangées se rejoignent au centre en une étoile à quatre branches, et c'est cette
ligne-là qui porte tout le défaut. `triangulate_surface`, lui, ne descend jamais
sous 0,42, sur aucune des onze formes : c'est la signature d'un Delaunay
raffiné, qui garantit un angle minimal et rien de plus. Il ne s'effondre jamais,
mais il ne monte jamais non plus.

### Ce qu'il faut retenir

**Le coût en mailles est le classement le plus stable** : sur les dix formes
rectilinéaires ou peu obliques, `grid_surface2` ≤ `grid_surface` <
`pave_surface` < `triangulate_surface`, avec un facteur quatre entre les
extrêmes à qualité au moins égale. Pour un calcul, c'est le facteur qui compte
juste après la qualité. Le cercle est la seule exception, et il l'est dans les
deux colonnes à la fois : `grid_surface2` y rend **plus** de mailles pour une
qualité effondrée, ce qui est la façon la plus nette de dire qu'il n'est pas
fait pour ça.

En pratique :

- **forme rectilinéaire** — `grid_surface2`, et d'autant plus si ses côtés
  n'ont pas été coupés aux angles qui leur font face ;
- **forme courbe** — `triangulate_surface` si la qualité du pire élément
  commande, `grid_surface` s'il faut des quadrangles ; **jamais
  `grid_surface2`**, qui y perd son cœur ;
- **forme franchement oblique** — `pave_surface`, dont les rangées épousent la
  pente ; mais tant qu'il reste des parois droites autour de l'oblique, les
  mailleurs en grille tiennent — la maison en est l'exemple ;
- **quadrangles obligatoires** — `pave_surface` accepte `all_quad=True` et
  refuse par une erreur claire un contour dont la parité l'interdit ;
- **triangles voulus** — `triangulate_surface`, seul à en produire par
  construction.

### Le front peut garder ses angles : `relax`

Les chiffres de `pave_surface` ci-dessus sont ceux du réglage par défaut. Entre
deux rangées, le front est relaxé — c'est ce qui l'empêche de se plisser — et
cette relaxation est un laplacien, qui **arrondit les angles**. Or un front ne
perd des nœuds qu'à ses coins : une fois les coins arrondis, il garde tous ses
nœuds pendant que son périmètre rétrécit, et le milieu du domaine sort plus fin
que la taille demandée. Un carré 20 × 20 à la taille 1 sort à **600 mailles au
lieu de 400**, les plus intérieures à 0,43 de l'aire visée.

`relax="along"` garde le même déplacement mais **projeté sur le front** :
l'écartement s'égalise encore, la forme n'est plus rabotée, et le carré sort
comme les 400 carrés exacts qu'on dessinerait. `relax="none"` ne relaxe pas du
tout. Aucun des trois ne gagne partout — sur une courbe, il n'y a pas d'angle à
préserver et le front a tout à gagner à se redresser :

| forme (taille 1) | `"free"` | `"along"` | `"none"` |
|---|---:|---:|---:|
| carré 20 × 20 | 600 mailles, pire 0,541 | **400, 1,000** | **400, 1,000** |
| L | 374, 0,315 | 287, 0,449 | 290, **0,564** |
| profil crénelé | 859, 0,240 | 620, **0,546** | **554**, 0,441 |
| bande étroite 40 × 3 | 581, 0,650 | 557, **0,662** | 559, 0,564 |
| cercle R = 10 | 569, 0,105 | 633, **0,257** | 639, 0,020 |

Le détail du mécanisme est sur la page
[Mailler](operateurs/maillage.md#relaxation-du-front--relax) des opérateurs. Les
deux mailleurs en grille prennent le même réglage, qui n'y gouverne que la
bande : la grille est posée droite quoi qu'il arrive.

### Deux limites connues

**Un vide que le contour ne borde pas n'est pas maillé en grille.**
`grid_surface2` prend l'écartement de ses lignes sur le contour ; là où le
contour ne dit rien — le triangle du toit de la maison, qu'aucune paroi
verticale ne borde — il coupe le vide en mailles entières tant qu'il ne dépasse
pas **trois fois** l'écartement moyen, et **l'abandonne au front au-delà**.
Remplir un tel vide de rangées ne crée pas du cœur : ces rangées n'existent que
pour être érodées par l'oblique qui les traverse, et elles morcellent le travail
du front au lieu de lui donner une région propre. Mesuré sur la maison :
l'abandon coûte **un** triangle là où le remplissage en coûtait cinq, et le 5ᵉ
centile monte de 0,644 à 0,676.

C'est aussi ce qui coûte le cercle, dont les vides valent quinze fois
l'écartement moyen. Aucun seuil ne sépare les deux cas — abandonner un vide de
huit moyennes abandonne a fortiori un vide de quinze — et c'est un choix
assumé : `grid_surface2` sert les formes rectilinéaires, `grid_surface` sert les
courbes.

**Le cercle est un objectif instable.** Sur une forme sans direction dominante,
déplacer la grille d'un millième change la pire maille du simple au décuple. Les
chiffres du cercle ci-dessus sont justes mais ne se prolongent pas : ne réglez
aucun paramètre dessus.

### Reproduire les figures

```bash
maturin develop --features extension-module,viz
python examples/comparer_mailleurs_2d.py

# Figures du livre :
PYRUCAST_IMG_DIR=book/src/img python examples/comparer_mailleurs_2d.py
```

Le script tient dans une fonction : elle prend un contour, le duplique trois
fois par `translate`, maille chaque copie par une méthode et réunit le tout sur
une figure.

```python
{{#include ../../examples/comparer_mailleurs_2d.py:comparer}}
```

## Mailler en 3D

À venir.
