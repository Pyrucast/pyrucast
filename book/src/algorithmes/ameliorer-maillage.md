# Améliorer un maillage existant : `regularize`, `cleanup`, `merge_triangles`

{{#pyalgo mesh.regularize mesh.cleanup mesh.merge_triangles}}

Les trois opérateurs font sur n'importe quel maillage ce que les paveurs font
déjà à leur propre ouvrage en fin de course. Ils sont **séparés et
composables** : l'ordre utile est en général triangles, puis topologie, puis
géométrie, mais rien ne l'impose.

## Deux marges, et une seule est de la géométrie

Un maillage peut être mauvais de deux façons sans rapport, et il ne faut pas les
confondre.

La **géométrie**, c'est où sont les nœuds, et c'est ce qu'un lissage corrige.
La **topologie**, c'est qui est voisin de qui, et **aucun lissage ne peut y
toucher** : un nœud intérieur entouré de trois quadrangles a des coins à 120° en
moyenne, et les déplacer n'y changera rien puisque les angles autour d'un nœud
somment à 2π quelles que soient les positions. D'où `cleanup` **à côté** de
`regularize` et non dedans.

## Le bord n'est jamais touché

Tout nœud porté par une arête de bord — une arête qu'une seule maille utilise —
est épinglé : il ne bouge pas, et rien ne l'abandonne. Le maillage garde donc
exactement le bord avec lequel il est entré, ce qui est la promesse que les
paveurs font déjà sur le contour qu'on leur donne.

## `regularize` — la géométrie

`angular=True` (défaut) utilise le lissage **angulaire** : chaque voisin `n`
propose la position obtenue en amenant `n → v` sur la bissectrice de l'angle que
`n` voit, à longueur inchangée, et on moyenne les propositions. Elle vise
l'angle droit là où le laplacien ne vise que le barycentre.

Deux garanties tiennent quelle que soit la règle, parce que le balayage est
celui des paveurs : **aucun nœud de bord ne bouge**, et une position n'est
retenue que si toutes les mailles incidentes restent valides *et* que la pire
qualité incidente ne baisse pas.

**Les deux se valent désormais**, et c'est mesuré. Sur le cercle pavé
(60 côtés, taille 0,05, 1 236 mailles) après 60 balayages :

| | qualité min | p1 | p5 |
|---|---|---|---|
| brut | 0,366 | 0,471 | 0,800 |
| laplacien | **0,806** | 0,807 | 0,863 |
| angulaire | 0,781 | 0,807 | **0,865** |

Les voilà à égalité au premier centile, le laplacien devant sur la pire maille
et l'angulaire devant au cinquième centile. L'angulaire dominait franchement
les trois colonnes tant que le maillage brut portait ses nœuds irréguliers :
depuis que `cleanup` sait effondrer l'étoile d'un nœud à trois mailles, il en
reste peu, et c'est précisément sur ceux-là que la règle angulaire prenait son
avantage. Les enchaîner — angulaire puis laplacien — reste ce qui marche le
mieux.

## `cleanup` — la topologie

Un **doublet** est un nœud intérieur n'ayant que deux quadrangles autour de lui,
qui partagent donc *deux* arêtes : le nœud est coincé dans un coin qu'aucun
lissage n'ouvrira. Une **valence fautive** est un nœud intérieur qui veut quatre
mailles et en a trois ou cinq.

Trois gestes. Le premier est purement topologique ; les deux autres ne peuvent
pas l'être, et c'est le cœur de l'affaire.

1. **Bascule de diagonale**, pour une valence trop forte : deux quadrangles
   partageant une arête forment un hexagone, qui se recoupe selon l'une de ses
   trois diagonales. Elle ne change ni nœud ni bord, et n'est appliquée que si
   elle fait strictement baisser l'erreur de valence en laissant les deux
   mailles convexes.
2. **Effondrement de l'étoile**, pour un nœud intérieur qui n'a que **trois**
   mailles autour de lui. Ni le doublet — qui en veut deux — ni la bascule —
   qui en veut deux quadrangulaires — ne l'atteignent.

Le second geste tient tout entier dans une identité. Autour d'un nœud portant
\\( q \\) quadrangles et \\( t \\) triangles, chaque quadrangle pose deux arêtes qui
ne le touchent pas et chaque triangle en pose une : l'**étoile** du nœud est
bordée par un polygone à

\\[ n = 2q + t \\]

côtés. Réciproquement, une décomposition d'un \\( n \\)-gone en \\( q\' \\)
quadrangles et \\( t\' \\) triangles **sans nœud intérieur** vérifie

\\[ 2q\' + t\' = n - 2. \\]

Avec \\( q + t = 3 \\), les deux se combinent en \\( 2q\' + t\' = 2q + t - 2 \\) : la
redécoupe existe toujours, et toujours avec **une maille de moins** que
l'étoile qu'elle remplace.

| \\( q, t \\) | bord de l'étoile | avant | après | mailles |
|---|---|---|---|---|
| 3, 0 | hexagone | 3 quadrangles | 2 quadrangles | 3 → 2 |
| 2, 1 | pentagone | 2 quadrangles, 1 triangle | 1 de chaque | 3 → 2 |
| 1, 2 | quadrangle | 1 quadrangle, 2 triangles | 1 quadrangle | 3 → 1 |
| 0, 3 | triangle | 3 triangles | 1 triangle | 3 → 1 |

Trois mailles autour d'un nœud intérieur, c'est une de trop quelles qu'elles
soient : les coins font 120° en moyenne et aucun lissage ne les redressera,
puisque les angles autour d'un nœud somment à 2π quelles que soient les
positions. C'est le seul geste de `cleanup` qui **supprime un nœud**, le seul
qui change le nombre de mailles, et le seul qui déplace quoi que ce soit. La
coupe est prise selon celle des rotations qui laisse le maillage au mieux, et
le geste s'applique si **l'un des deux** progresse : l'erreur de valence baisse
strictement, ou la pire des nouvelles mailles bat la pire des anciennes. Il est
refusé net si une maille sortirait retournée.

### Pourquoi il faut relaxer avant de juger

Supprimer un nœud **étire mécaniquement** l'anneau qui restait autour : les six
nœuds de l'hexagone ne bougent pas, et deux mailles s'y logent là où il y en
avait trois. Mesurée sur-le-champ, la redécoupe paraît donc presque toujours
plus mauvaise que l'étoile qu'elle remplace — alors qu'elle sera bonne dès que
les nœuds auront été relâchés, ce qui arrive toujours : les paveurs lissent
après chaque rangée, et `regularize` est l'étape suivante de l'appelant.

C'est la différence de fond avec la bascule de diagonale, qui, elle, **ne
déplace aucun nœud** : ce qu'elle mesure est définitif, et elle peut s'imposer
un plancher de qualité immédiat (`QUALITY_FLOOR`, 70 % de ce qui était là) sans
rien s'interdire d'utile. Le même plancher appliqué à l'effondrement supprime
**53 gestes utiles sur 61** — mesuré.

D'où l'ordre retenu : appliquer la découpe, **relaxer l'anneau**, mesurer, et
défaire entièrement si la pire maille du voisinage a baissé. Et lorsque le
geste est gardé, la relaxation est gardée avec lui : juger sur des positions
qu'on rejetterait ensuite reviendrait à mesurer un maillage que personne ne
reçoit — c'est mesuré aussi, et cela coûtait une maille à 0,055 sur la maison.

Un dernier garde-fou tient l'entrée : rien n'est tenté si le geste n'apporte
**ni** valence **ni** forme. C'est que le verdict ci-dessus juge le *voisinage*,
donc localement, et ne voit pas qu'une maille médiocre ailleurs vient de
devenir la pire du maillage. Le retirer fait gagner quelques nœuds réguliers de
plus et perdre la garantie de non-régression sur la pire maille — l'échange est
mauvais : une pire maille qui recule casse un calcul, treize nœuds irréguliers
de plus ne se voient nulle part.

**Aucun nœud du contour ne bouge**, et aucun n'est abandonné : c'est la
garantie qui tient, la même que pour les trois opérateurs, et celle avec
laquelle raisonner. « Rien ne bouge du tout » n'en est pas une : `cleanup`
déplace les nœuds des anneaux qu'il a effondrés, et seulement ceux-là. Partout
ailleurs il rend vos nœuds tels quels, et le maillage que vous lui donnez n'est
jamais modifié.

Les deux dernières lignes du tableau perdent un triangle chacune — deux d'un
coup, donc la parité qui lie leur nombre aux arêtes de bord (voir
`merge_triangles` ci-dessous) reste intacte.

### La paire, que le geste précédent ne peut pas atteindre

Deux nœuds intérieurs **reliés par une arête**, dont l'un au moins manque d'une
maille, forment un motif qu'aucun des gestes ci-dessus ne prend. Abandonner
l'un des deux tout seul ferait tomber deux de ses voisins de quatre à trois :
un nœud irrégulier échangé contre deux, et l'effondrement le refuse à juste
titre.

Pris ensemble, le calcul s'inverse. Leurs étoiles se recouvrent sur les mailles
qui portent l'arête commune, si bien qu'à eux deux ils n'en portent que
\\( \mathrm{val}(a) + \mathrm{val}(b) - 2 \\) — et la même identité
\\( 2q\' + t\' = n - 2 \\) recoupe ce qui les borde :

| \\( \mathrm{val}(a), \mathrm{val}(b) \\) | bord | avant | après | mailles |
|---|---|---|---|---|
| 3, 3 | hexagone | 4 quadrangles | 2 quadrangles | 4 → 2 |
| 3, 4 | heptagone | 4 quadrangles, 1 triangle | 2 quadrangles, 1 triangle | 5 → 3 |

Le geste abandonne donc **deux nœuds et deux mailles d'un coup**. Un triangle
présent dans l'étoile est **repris**, jamais créé ni perdu : la parité qui lie
leur nombre au bord interdit d'en faire apparaître un, et la redécoupe de
l'heptagone réclame précisément celui qui était là.

L'arithmétique de valence le justifie : les quatre mailles offrent seize coins
et les deux qui les remplacent en offrent huit ; la paire en emporte six avec
elle, l'anneau en rend deux, et les deux erreurs de la paire sont annulées.
Le gain est de **+2** sur une paire ordinaire et monte à **+4** quand l'anneau
porte un nœud de valence 5 à dépenser.

La paire est examinée **avant** le nœud seul : c'est le motif le plus
spécifique et la meilleure affaire. Dans l'autre ordre, l'effondrement d'un
nœud prend l'un des deux et la paire n'a jamais sa chance — mesuré.

Comme tout geste qui supprime un nœud, il se juge après relaxation : la qualité
mesurée juste après la découpe *baisse* (0,535 à 0,485 en médiane), et la lire
là conduirait à refuser précisément les gestes qui paient.

Sur un maillage frais sorti d'un paveur, `cleanup` ne trouve **rien à faire** —
le paveur passe la même main en fin de course. Son intérêt est ailleurs : un
maillage lu depuis gmsh, ou sorti de `merge_triangles`.

### Le sens de parcours n'a pas à vous préoccuper

Toutes les mesures de qualité employées ici sont **signées** : une maille lue
en sens horaire compte négatif, ce qui se lit comme *retournée*. C'est
volontaire — c'est ce signe qui empêche le lissage de retourner une maille — et
un maillage entièrement horaire n'a pourtant rien d'anormal : un paveur rend le
sens du contour qu'on lui a donné, si bien qu'un domaine maillé depuis un
contour extérieur inversé (`invert`) sort horaire.

Les trois opérateurs remettent donc chaque maille en sens trigonométrique à la
lecture et rendent le maillage dans le sens où il est entré. Rien à faire de
votre côté, et `orient` n'est pas un préalable.

## `merge_triangles` — les triangles, par paires

En comptant les côtés de chaque face, \\( 4Q + 3T = 2E_\text{int} + E_\text{bord} \\),
donc \\( T \equiv E_\text{bord} \pmod 2 \\) : **le nombre de triangles a la
parité du nombre d'arêtes de bord**, qui est intouchable. Aucune suite
d'opérations locales ne peut la changer, et un maillage à nombre impair de
triangles en garde un.

C'est aussi pourquoi « effondrer un triangle en un nœud » ne marche pas :
fusionner ses trois coins en un seul coûte une arête à chaque quadrangle
voisin, qui devient triangle à son tour. Gain net nul.

Trois gestes, tous purement topologiques — aucun nœud ne bouge, aucun n'est
créé :

1. **Fusion.** Deux triangles partageant une arête font un quadrangle. C'est à
   prendre ou à laisser : il n'y a qu'un quadrangle possible, et s'il sort
   rentrant la paire est bloquée.
2. **Regroupement.** En prenant un quadrangle voisin avec eux, les trois mailles
   font un **hexagone** — 3 + 3 + 4 arêtes moins deux fois les deux partagées —
   qui se coupe en deux quadrangles selon l'une de ses trois grandes diagonales.
   Là où la fusion n'offre qu'un candidat — à prendre ou à laisser — celui-ci
   en offre trois, et c'est ce qui débloque le gros des paires. Prendre **deux**
   quadrangles plutôt qu'un couvre en plus la configuration en éventail autour
   d'un nœud, et en retire encore quelques-uns sur une ellipse.

   Trois quadrangles donneraient un octogone, qui veut trois quadrangles et
   bien plus de façons de le couper. La limite est mise à deux : le rendement
   s'essouffle, pas la recherche.
3. **Marche.** Un triangle et un quadrangle voisin font un **pentagone**, qui se
   redécoupe en quadrangle + triangle de cinq façons : le triangle avance d'une
   maille et va chercher un partenaire plus loin.

Le contour du motif recoupé n'est jamais modifié, dans les trois cas : ces
gestes **ne peuvent pas** changer le bord du maillage, par construction.

La validité ne se négocie pas, ici comme dans les paveurs : deux triangles dont
l'union n'est pas convexe restent deux triangles. **Rappelez l'opérateur après
un lissage** — une fusion refusée le devient souvent une fois les nœuds
déplacés.

## Ce que la chaîne donne

Cercle pavé par `grid_surface`, 1 236 mailles dont 8 triangles :

| | mailles | triangles | qualité min | p1 | p5 |
|---|---|---|---|---|---|
| brut | 1 236 | 8 | 0,366 | 0,471 | 0,800 |
| après la chaîne | 1 216 | **0** | **0,781** | **0,803** | **0,841** |

Les triangles tombent à **zéro** — une seule passe de `merge_triangles` y
suffit sur ce cercle — et le **pire** quadrangle passe de 0,366 à 0,781.
**Chaque maillage intermédiaire reste valide**, et le tour répété converge au
lieu de dériver.

Un point à connaître : la sortie brute a la queue basse plus mince qu'avant que
`cleanup` sache effondrer une étoile à trois mailles — premier centile à 0,471
contre 0,643. C'est le prix du geste, qui supprime un nœud et laisse ses voisins
un peu étirés là où il passe, le temps que le lissage suivant les reprenne. Il
est largement rendu : trois fois moins de triangles à la sortie du paveur, et
aucun à l'arrivée.
