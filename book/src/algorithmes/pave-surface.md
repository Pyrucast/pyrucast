# Pavage frontal d'un contour fermé : `pave_surface`

{{#pyalgo mesh.pave_surface}}

`pave_surface(contour, element_type, size=None, prefer_quads=False, relax="free")`
remplit le même contour que `triangulate_surface`, mais en **posant directement
des quadrangles**, par rangées qui avancent depuis le bord vers l'intérieur.
C'est
la version quadrangle de l'opérateur Cast3M `SURF`.

## Pourquoi un second mailleur de surface

`triangulate_surface` accepte `"QUA4"`, mais il triangule d'abord et
**recombine** ensuite les triangles deux par deux : ce qu'on obtient dépend de
la chance des appariements, les valences sont désordonnées et rien n'est
aligné sur le bord. `pave_surface` ne recombine pas. Il pose des rangées
**parallèles au contour**, ce qui est précisément la structure recherchée en
éléments finis, où les gradients de contrainte et de flux sont les plus forts
près des frontières.

| | `triangulate_surface` | `pave_surface` |
|---|---|---|
| méthode | Delaunay contraint + Ruppert | front avançant |
| élément naturel | `TRI3` | `QUA4` |
| `QUA4` obtenu par | recombinaison de paires | construction |
| rangées alignées sur le bord | non | oui |
| quadrangles seuls | impossible | préférés (`prefer_quads=True`), sans garantie |

## Convention

Identique à `triangulate_surface`, et volontairement : les deux opérateurs
partagent leur lecture de contour. `contour` est un `Mesh` d'une ou plusieurs
**boucles `SEG2` fermées**, chacune dans **un seul** sous-maillage
(`pyrucast.mesh.consolidate`), orientées par l'appelant — **CCW** pour une frontière
extérieure, **CW** pour un trou. Plusieurs boucles CCW disjointes pavent
plusieurs domaines indépendants en une passe. La configuration peut être en
dimension **2**, ou une boucle **plane en 3D** (ajustée à son plan de meilleur
approximation par la méthode de Newell, pavée dans ce plan, puis relevée).

**Le contour est figé** : les nœuds d'entrée sont réutilisés tels quels (mêmes
identifiants, mêmes positions), ne sont jamais déplacés, et aucun nœud n'est
ajouté sur une arête de bord. Voir *Le contour est intouchable* ci-dessous.

`element_type` vaut `"QUA4"`, `"QUA8"` ou `"QUA9"` (les formes quadratiques
sont dérivées du maillage `QUA4`). `size` fixe la longueur d'arête visée ; par
défaut, la longueur moyenne des segments de bord du domaine.

## Méthode

Le **front** part du bord du domaine et avance vers l'intérieur. Il est
toujours un ensemble de boucles simples et disjointes, matière à gauche ; cet
invariant n'est jamais supposé, il est **maintenu**.

1. **Une rangée par tour.** À chaque nœud du front, le nombre de quadrangles
   voulus est \\( k = \operatorname{round}(\theta / 90°) \\), borné à
   \\( 1..4 \\), où \\( \theta \\) est l'angle intérieur. Ce n'est pas un seuil
   réglé : si \\( k \\) quadrangles entourent le nœud, ses voisins forment un
   chemin de \\( k+1 \\) sommets, dont \\( k-1 \\) sont neufs — vouloir des
   angles droits fixe \\( k \\). Les nouveaux nœuds se placent sur les rayons
   qui découpent le secteur en \\( k \\) parts égales.
2. **Refus et retrait.** Un quadrangle non strictement convexe a un jacobien
   négatif à son coin rentrant : aucun code éléments finis ne peut l'intégrer.
   Une rangée qui en produirait un, dont les arêtes croiseraient le front, ou
   qui laisserait sa boucle **retournée**, est **refusée** ; le planificateur
   dit *quels* nœuds sont en cause, et la rangée est reprise moins loin à cet
   endroit seulement. Le retournement est le cas subtil : une rangée consomme
   de la matière, elle ne peut donc que rétrécir la région bornée par sa
   boucle, jamais en inverser le sens. Celle qui le fait est celle posée sur
   une **lèvre** — le résidu que deux fronts se rencontrant de face laissent
   entre eux, déjà plus mince que la maille voulue. Ses quadrangles passent
   tous les tests locaux, sa chaîne ne croise rien, et la boucle ressort à
   l'envers ; à partir de là chaque rangée suivante la *gonfle* au lieu de la
   réduire, jusqu'à sortir de la matière et entraîner avec elle toute couture
   rencontrée. Le signe de l'aire orientée distingue les deux cas sans
   ambiguïté — une boucle extérieure décroît vers zéro par le haut, le front
   d'un trou s'en éloigne par le bas, et chacune garde son signe sa vie
   durant. La lèvre refusée est laissée à la fermeture, qui la remplit ou la
   soude sur place.
3. **Détente.** Une rangée fraîchement posée garde les coudes que le placement
   sur les bissectrices lui a laissés, et ils se composent : un nœud à 216°
   offre à ses voisins un secteur qu'aucun gabarit ne remplit bien, la rangée
   suivante est refusée et la boucle cale. Le front est donc relâché **le long
   de lui-même** — un nœud n'a de voisins engagés que derrière lui, un laplacien
   ordinaire le ramènerait d'où il vient. Chaque déplacement est pesé sur les
   quadrangles que le nœud porte déjà ; mais ceux-ci sont tous *derrière* lui et
   ne voient rien de la ligne devant, si bien que deux portions du front peuvent
   glisser l'une sur l'autre sans qu'aucun test local ne bronche. La boucle est
   donc **aussi interrogée sur sa simplicité**, et une détente qui la lui coûte
   est reprise en entier. Sans cela, un cercle à vingt côtés maillé au cinquième
   de son rayon rendait une boucle croisée, donc irremplissable, et le mailleur
   abandonnait ; il le pave. Mesuré une fois par appel plutôt qu'une fois par
   passe : **+14 %** contre +31 % sur un maillage que le front pose seul, pour
   les mêmes cas sauvés — et gratuit dès qu'une grille fait le travail.
4. **Couture.** Deux nœuds de front qui se rapprochent à moins d'environ une
   demi-maille sont **identifiés**. La même opération *scinde* une boucle
   quand les deux nœuds lui appartiennent — c'est ainsi qu'une géométrie
   concave se divise — et *joint* deux boucles sinon — c'est ainsi qu'un trou
   est absorbé. Les trous n'ont donc aucun traitement particulier.

   Identifier réécrit sur le survivant **toutes** les mailles qui utilisaient
   l'autre nœud, triangles compris : un triangle resté accroché au nœud
   abandonné pendant que ses voisins déménagent est une maille pendue à un
   nœud que plus rien ne touche, soit trois fissures d'un coup. Et le
   survivant peut être un nœud du contour — il doit l'être dès que l'autre en
   est un, puisqu'un nœud du contour ne s'abandonne jamais. Deux coutures
   successives peuvent alors poser les **deux bouts d'une même arête** sur
   deux nœuds du contour qui en ont un troisième entre eux : cette arête
   longe le bord en enjambant un nœud, et la lentille d'aire nulle qui reste
   n'est refermable par aucune soudure, tous ses nœuds étant intouchables.
   Elle est **recousue** : la maille qui porte la corde reprend en elle les
   nœuds qu'elle a sautés. Elle ne grossit de rien, la lentille ne couvrant
   rien, et le bord redevient entier.
5. **Déblocage.** Une boucle qui n'avance plus est **coupée en deux** par une
   corde, et les deux moitiés reprennent le pavage.
6. **Fermeture.** Une boucle réduite à six nœuds ou moins est remplie par
   décomposition, sans jamais découper une arête (ce qui laisserait un nœud
   en T, donc un maillage non conforme). Encore faut-il qu'elle soit
   remplissable : une boucle qui **se croise** ne l'est pas, ses deux lobes
   tournant en sens contraire, et toute décomposition en laisse au moins un
   morceau **retourné**. Le signe de l'aire ne suffit pas à les repérer — quand
   les deux lobes se valent, l'aire qu'ils enferment s'annule presque et le
   signe qui reste est celui du plus gros, un accident. La boucle est donc aussi
   interrogée sur sa **simplicité** ; si elle est mince, elle est soudée comme
   une lèvre. C'est ce qui manquait : sur un cercle, deux mailles ressortaient
   à jacobien négatif.
7. **Nettoyage topologique**, puis **lissage** sous garde de validité qui ne
   déplace jamais un nœud du contour. Dans cet ordre : lisser un nœud qui n'a
   pas le bon nombre de mailles autour de lui ne fait qu'étaler l'erreur sur
   ses voisins.
8. **Refus final.** Chaque maille est mesurée : aucune ne sort d'aire nulle ou
   négative. Toutes les étapes ci-dessus ont déjà leur garde, exacte et locale ;
   celle-ci est le filet sous toutes les autres, et ce qu'elle attrape est une
   garde qui a laissé passer quelque chose. Le maillage n'est alors **pas
   rendu** : l'opérateur sort en erreur, en situant la pire maille. Une maille
   retournée a un jacobien négatif et une maille plate n'en a pas ; ni l'une ni
   l'autre n'est intégrable, donc un maillage qui en porte une n'est pas un
   maillage médiocre mais un maillage faux. **Mieux vaut une erreur qu'un
   maillage faux** — c'est la règle, et elle vaut pour tous les mailleurs.

Le nettoyage corrige ce que le lissage ne peut pas atteindre, parce que c'est
de la **connectivité** et non de la géométrie :

- un **doublet** — un nœud intérieur n'ayant que deux mailles autour de lui,
  qui partagent donc *deux* arêtes. Il reste coincé dans un coin quelles que
  soient les positions ; fusionner les deux mailles supprime le nœud et le coin
  d'un coup ;
- une **valence fautive**. Un nœud intérieur veut quatre mailles : avec trois,
  les angles valent 120° en moyenne, avec cinq, 72°, et aucun lissage n'y peut
  rien puisque les angles autour d'un nœud somment à 2π quelles que soient les
  positions. Deux mailles voisines forment un hexagone, qui se recoupe selon
  l'une de ses trois diagonales ; changer de diagonale déplace une unité de
  valence. C'est le seul geste, il ne change ni le nombre de nœuds ni le bord,
  et il n'est appliqué que s'il fait strictement baisser l'erreur de valence.

Le nombre de mailles voulu à un nœud est le même
\\( \operatorname{round}(\theta / 90°) \\) que dans la classification des
rangées, avec ici \\( \theta \\) la somme des angles incidents : \\( 2\pi \\) à
l'intérieur — d'où le quatre familier — et moins au bord, ce qui donne trois le
long d'une arête droite et deux dans un coin droit. Une seule formule, aucun
cas particulier « nœud de bord ».

Toutes les décisions topologiques — convexité, croisement de segments —
passent par le prédicat **exact** `orient2d` (technique de Shewchuk, partagé
avec le mailleur volumique). Ce ne sont donc pas des estimations.

## Le contour est intouchable

Tous les nœuds du contour reviennent dans le maillage, à leur position, et
**aucun nœud n'est jamais ajouté sur une arête de bord**. La discrétisation du
bord appartient à l'appelant : elle porte en général les conditions aux
limites, et un nœud glissé au milieu d'un segment serait un nœud que personne
n'a demandé. Rien dans le paveur ne découpe une arête de bord, et la couture —
la seule opération qui abandonne un nœud — refuse d'abandonner un nœud de
contour.

Corollaire : un contour avec lequel le paveur ne peut pas travailler est
**signalé**, pas contourné. Deux cas, tous deux renvoyés en erreur nommant le
problème :

- `prefer_quads` sur une boucle à nombre **impair** de segments. Un polygone à
  nombre impair de côtés n'admet aucun remplissage en quadrangles seuls ; le
  pavage ne peut pas changer cette parité — une rangée la conserve, une couture
  retire deux nœuds — et rééquilibrer le compte reviendrait à ajouter un nœud
  au bord ;
- un contour si grossier, ou si irrégulier pour la taille demandée, que le
  front **se replie sur lui-même** en laissant une région impossible à
  remplir. L'erreur indique l'endroit.

## Préférer les quadrangles : `prefer_quads`

Laissé à lui-même (`prefer_quads=False`), un contour impair coûte simplement **un**
triangle, rendu dans un sous-maillage `TRI3` séparé — avec les quelques mailles
qu'un polygone résiduel trop déformé n'a pas pu rendre carrées : la fermeture
préfère deux triangles à une maille de jacobien négatif, et la validité n'est
jamais échangée.

Avec `prefer_quads=True`, la parité devient une exigence sur l'entrée :
discrétisez chaque boucle de bord avec un nombre pair de segments — c'est votre
donnée, une boucle impaire est refusée. Le pavage tient ensuite le front pair
jusqu'au bout : chaque rangée, chaque couture, chaque découpe garde des boucles
paires. Ce qu'il ne fait pas disparaître, ce sont les triangles **de forme** :
un polygone résiduel trop déformé pour se couper en quadrangles valides est
encore fermé par des triangles, puisque la validité passe avant. D'où le nom :
une **préférence**, pas une garantie.

## Relaxation du front : `relax`

Après chaque rangée, la chaîne fraîchement posée est **relaxée** — c'est ce qui
empêche le front de se plisser. Cette relaxation est un laplacien, et un
laplacien **arrondit les angles**. Cela coûte plus qu'il n'y paraît : un front
ne perd des nœuds qu'aux endroits où son angle intérieur en demande moins de
deux, c'est-à-dire à ses **coins**. Une fois les coins arrondis, il garde tous
ses nœuds pendant que son périmètre rétrécit, son écartement se dégrade rangée
après rangée, et le milieu du domaine sort plus fin que la taille demandée.

Le carré 20 × 20 à la taille 1 le montre sans ambiguïté : il contient 400
mailles unitaires, et le paveur devrait les poser. Ses quatre coins sont des
*fins de rangée*, et c'est une fin de rangée qui fait perdre à la rangée les
huit nœuds que le périmètre perd.

| `relax` | mailles | pire maille | aire médiane |
|---|---:|---:|---:|
| `"free"` (défaut) | 592 | 0,619 | 0,62 |
| `"along"` | **400** | **1,000** | **1,00** |
| `"none"` | **400** | **1,000** | **1,00** |

- **`"free"`** — le nœud va où le laplacien le pousse. C'est le comportement
  historique, et le seul qui ne laisse jamais le front se plisser.
- **`"along"`** — le même déplacement, **projeté sur le front** : l'écartement
  s'égalise encore, la forme n'est plus rabotée.
- **`"none"`** — le front reste exactement où la rangée l'a posé.

Il n'y a pas de bonne réponse universelle, et c'est pourquoi c'est un choix.
`"along"` et `"none"` gagnent sur tout ce qui a des angles à garder ; sur une
courbe, il n'y a rien à préserver et tout à redresser. Mesuré sur un carré et
une bande à la taille 1, puis sur trois formes de
[Mailler une géométrie](../mailler.md), à leur taille :

| forme | `"free"` | `"along"` | `"none"` |
|---|---:|---:|---:|
| carré 20 × 20 | 592 mailles, pire 0,619 | **400, 1,000** | **400, 1,000** |
| bande étroite 40 × 3 | 123, 0,702 | **120, 1,000** | 123, 0,684 |
| L à cotes quelconques | 111, 0,680 | **66, 0,780** | **66**, 0,774 |
| créneau, base coupée | 816, 0,118 | 586, 0,426 | **577, 0,546** |
| cercle R = 1 | 547, 0,081 | **610, 0,281** | 620, 0,027 |

Le cercle ne se laisse départager par aucun réglage : selon la taille demandée,
chacun des trois passe devant ou tombe près de zéro (de 0,001 à 0,281, voir
[Mailler une géométrie](../mailler.md)). **Sur une courbe, ne comptez pas sur
`relax` : prenez `triangulate_surface` ou `grid_surface`.**

Un front qui n'arrive pas à converger ne s'obstine pas : le pavage s'arrête sur
une erreur qui le dit, plutôt que de rendre un maillage fait de ce qui restait.

## Exemple Python

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-pave_surface}}
```

## Interruption

Le pavage interroge les signaux Python entre deux rangées : `Ctrl+C` pendant
un maillage long lève `KeyboardInterrupt`. Côté Rust, la forme
`pave_surface_cancellable(..., cancel)` prend un jeton `Cancel`.

## Qualité

La qualité d'une maille est la **mean ratio** de son pire coin,
\\( 2(a \\times b) / (|a|^2 + |b|^2) \\), qui ne vaut 1 que si l'angle est droit
**et** les deux arêtes égales. Pas le seul sinus de l'angle : celui-ci donne
1,000 à un rectangle 10:1, et laisse donc la garde du lissage écraser une
maille tant qu'elle garde ses coins droits.

Sur une plaque percée, le cœur du maillage est fait de rectangles à angle
droit et de valence régulière — aucune maille inversée, angle médian à
l'équerre, très large majorité de nœuds intérieurs à quatre mailles — ce qui
est exactement ce qu'on demande à un maillage quadrangulaire.

La faiblesse résiduelle est l'**élancement** : les mailles sont d'équerre mais
sensiblement plus longues que larges. C'est précisément ce que la mean ratio
fait apparaître là où le sinus le cachait, puisqu'un rectangle 10:1 obtient un
jacobien normalisé de 1,000 et une mean ratio bien inférieure. Les deux
mesures disent la même chose de la forme ; une seule des deux la voit.

## Coût

Le coût est **essentiellement linéaire** en nombre de mailles, sur plusieurs
ordres de grandeur, tant que le front avance sans se coincer : le front croît
comme la racine du nombre de mailles, et l'index spatial est reconstruit à
chaque rangée pour ce prix-là. Il augmente nettement quand les blocages se
multiplient, la boucle repassant alors par les cordes de déblocage et les
fermetures.

Le temps se répartit en gros en trois tiers : la pose des rangées, le nettoyage
topologique et le lissage final. `triangulate_surface` va plus vite sur la même
géométrie, mais laisse une part notable de triangles en `QUA4`.

## Pièges

- **Une boucle par sous-maillage.** Comme pour `triangulate_surface`, une
  boucle fermée doit tenir dans un seul sous-maillage : `pyrucast.mesh.consolidate`
  après avoir uni les côtés.
- **Orientation.** Un trou doit être **CW**. `pyrucast.mesh.invert` retourne
  un cercle construit en CCW.
- **La taille du contour compte.** Le front part de la discrétisation du bord
  et converge vers `size` en quelques rangées. Un contour beaucoup plus
  grossier que `size` donne donc des premières rangées plus grosses que
  demandé.

## Limitations actuelles

- La taille des mailles n'est pas uniforme : l'espacement le long du front et
  la distance d'avance ne sont pas asservis l'un à l'autre, d'où un élancement
  médian proche de 2 et une taille d'arête étalée d'un facteur 10. Les angles,
  eux, restent droits.
- Deux fronts qui se rejoignent de face laissent normalement un éclat de
  recouvrement, dégénéré et sans matière : il est écarté. Au-delà d'une
  demi-maille d'aire, en revanche, c'est une région perdue, et le paveur
  **sort en erreur** en indiquant l'endroit plutôt que de rendre un maillage
  troué.
- **Front convexe sans coin.** Un front ne perd des nœuds que par couture, et
  une couture n'est acceptée que si elle laisse toutes les mailles valides. Un
  contour circulaire n'a aucun coin : son front garde donc son nombre de nœuds
  pendant qu'il se contracte, ses arêtes raccourcissent, et les rangées
  finissent par ne plus tenir. Le paveur s'en sort par des cordes de
  découpage, mais le débit s'effondre — de l'ordre de 10³ mailles/s sur un
  disque finement discrétisé, contre 10⁵ sur la plaque trouée. Une géométrie
  comportant des coins, ou un contour discrétisé près de la taille visée, ne
  rencontre pas ce cas.
- Pas de champ de taille variable : `size` est uniforme par domaine.
- **Un bras étroit n'a pas d'intérieur.** C'est la forme aiguë du défaut
  précédent. Dans un bras de quatre ou cinq mailles de large — une tour de
  créneau, une nervure — le front avance d'une maille depuis chaque paroi et
  les deux se rencontrent après deux rangées : il ne reste aucune place où
  poser des mailles franches, et toute la largeur est ligne de collision. La
  première couche épouse bien la paroi, mais elle est déjà écrasée par sa
  jumelle d'en face. Sur le profil crénelé, les tours sortent à un élancement
  médian de 1,32 contre 1,11 pour `grid_surface`, et le maillage entier coûte
  407 mailles contre 271. Là encore, pour une forme faite de bras
  rectilinéaires, c'est `grid_surface` qu'il faut.
- **Un rectangle ne donne pas une grille.** C'est la limite de fond, et elle
  n'est pas réparable dans un paveur frontal : quatre fronts partis des quatre
  côtés se rencontrent en quatre coutures diagonales convergeant vers deux
  points singuliers. Sur un rectangle 0,6 × 0,3 à 0,02, le paveur rend 650
  mailles et 6 triangles là où la grille en demande 450. Pour une forme
  rectilinéaire, c'est `grid_surface` qu'il faut.
