# Mailleur volumique : `triangulate_volume`

{{#pyalgo mesh.triangulate_volume}}

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-triangulate_volume}}
```

Le **compagnon 3D** de `triangulate_surface` : il remplit l'intérieur d'une
**enveloppe fermée en TRI3** avec des `TET4`. Les
normales de l'enveloppe doivent pointer **vers l'extérieur de la matière** ;
une forme concave est admise, et une cavité interne n'est qu'une autre
surface fermée dont les normales pointent vers le trou — elle se soustrait
d'elle-même, sans argument dédié.

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:triangulate_volume_taille}}
```

L'enveloppe est **respectée exactement** : ses nœuds sont réutilisés tels
quels (mêmes `NodeId`, mêmes positions), et aucun nœud n'est posé sur la
surface — les nœuds ajoutés le sont strictement à l'intérieur. Ce n'est pas
un effort au mieux : avant d'écrire quoi que ce soit, l'opérateur vérifie que
le bord du maillage produit est *exactement* l'ensemble des facettes reçues.

## Ce qui se passe, et pourquoi

Le mailleur enchaîne cinq étapes. Chacune répond à une difficulté précise, et
il vaut la peine de savoir laquelle, parce que les messages d'erreur les
nomment.

### 1. Prédicats exacts

Tout repose sur deux questions posées des millions de fois : *de quel côté du
plan `(a, b, c)` se trouve `d` ?* et *le point `e` est-il dans la sphère
passant par `a, b, c, d` ?* Ce sont les signes de deux déterminants :

\\[
\mathrm{orient3d}(a,b,c,d)=\begin{vmatrix}
b_x-a_x & b_y-a_y & b_z-a_z\\\\
c_x-a_x & c_y-a_y & c_z-a_z\\\\
d_x-a_x & d_y-a_y & d_z-a_z
\end{vmatrix}
\qquad
\mathrm{insphere}(a,b,c,d,e)=-\begin{vmatrix}
a_x-e_x & a_y-e_y & a_z-e_z & \lVert a-e\rVert^2\\\\
b_x-e_x & b_y-e_y & b_z-e_z & \lVert b-e\rVert^2\\\\
c_x-e_x & c_y-e_y & c_z-e_z & \lVert c-e\rVert^2\\\\
d_x-e_x & d_y-e_y & d_z-e_z & \lVert d-e\rVert^2
\end{vmatrix}
\\]

`orient3d` vaut six fois le volume signé, et il est **positif** exactement
quand le tétraèdre `(a,b,c,d)` est bien orienté au sens de `TET4` — face
`0-1-2` vue en sens direct depuis le nœud 3.

Ce qui compte n'est pas la précision de ces valeurs mais leur **cohérence** :
si `orient3d(a,b,c,d)` répond « au-dessus », alors `orient3d(b,a,c,d)` doit
répondre « en dessous », et un point ne peut pas être à la fois dans un
tétraèdre et hors de ses quatre faces. En `f64` nu, près d'une
dégénérescence, cette cohérence tombe — et une seule réponse contradictoire
corrompt le graphe d'adjacence. C'est de cette façon qu'un mailleur
incrémental tourne en boucle ou produit des cellules qui se recouvrent.

Ni une tolérance ni un jitter n'y remédient : ils rendent le prédicat
*généralement* juste, pas cohérent avec lui-même. `triangulate_volume` calcule donc
le **signe exact**, par la technique des expansions flottantes de Shewchuk :
une estimation en `f64` comparée à une borne d'erreur rigoureuse, puis, dans
le seul cas où le signe reste indécidable, une réévaluation en arithmétique
exacte. Un cube, une grille régulière, des coins cosphériques sont ainsi
**décidés** et non devinés.

### 2. Triangulation de Delaunay des nœuds de l'enveloppe

Construite point par point selon **Bowyer-Watson** : on supprime tous les
tétraèdres dont la sphère circonscrite contient le nouveau point `p`, puis on
rebouche la cavité en joignant `p` à chaque face de son bord. L'appartenance
à la cavité est `insphere > 0` et rien d'autre, ce qui garantit que la cavité
reste **étoilée** — visible en entier depuis `p` — et donc que le
rebouchage produit des cellules bien formées.

Les points cosphériques ne sont pas perturbés : `insphere == 0` signifie
simplement « hors cavité ». On obtient l'une des triangulations de Delaunay
valides de la configuration dégénérée, choisie de façon cohérente.

### 3. Récupération du bord

La triangulation précédente pave l'**enveloppe convexe** des nœuds et ne doit
rien à la surface d'où ils viennent : une arête de l'enveloppe peut être
traversée par un tétraèdre, une facette percée par une arête. Avant de
pouvoir distinguer l'intérieur de l'extérieur, chacune doit *apparaître*
dans la triangulation.

C'est la partie difficile, et pour une raison de fond : **certains polyèdres
n'admettent aucune tétraédrisation sur leurs propres sommets**. Le prisme
tordu de Schönhardt est l'exemple d'école. Plus près de vous : sur les 64
façons de trianguler le bord d'un cube, la plupart ne se remplissent pas. Ce
n'est donc pas un algorithme perfectible, c'est une question dont la réponse
est parfois « il n'y en a pas ».

La récupération procède obstacle par obstacle — bascules locales, puis
reconstruction de la poche qui bloque — et, quand elle n'y arrive pas, elle
nomme l'arête ou la facette en cause plutôt que de rendre un maillage qui ne
correspond pas à la surface reçue.

**Reconstruire une poche : deux remplisseurs.** Le premier *cherche*, en
faisant croître un pavage cellule par cellule depuis la surface de la poche.
Il est complet — lui seul peut prouver qu'*aucun* remplissage n'existe — mais
il est exponentiel et s'arrête à quelques cellules. Le second ne cherche pas :
il **calcule** l'unique candidat canonique, la triangulation de Delaunay des
sommets de la poche, et lui pose une seule question — contient-elle chaque
face de la surface de la poche ? Si oui, les cellules qui tombent à
l'intérieur pavent exactement la poche, pour le prix d'une triangulation quel
que soit son volume ; si non, il **nomme la face** sur laquelle il a buté.

Ce nom est une instruction, pas un diagnostic : en absorbant la cellule située
de l'autre côté de cette face, celle-ci cesse d'être sur la surface de la
poche, et l'obstacle ne peut pas se représenter. On repose alors une question
strictement plus grande, sans arbre de recherche ni retour arrière. C'est ce
qui rend abordable la récupération d'une facette prise dans un plan interne —
la diagonale d'un quadrilatère de paroi d'extrusion, par exemple — là où la
recherche exhaustive renonçait. La facette à récupérer est passée comme un
**mur à deux faces** : la poche est coupée en deux le long d'elle et chaque
moitié est remplie depuis son propre côté, de sorte qu'une facette contenue
dans la triangulation est une facette récupérée.

> **Ce que cela ne fait pas.** Une triangulation de Delaunay est ce qu'elle
> est : on ne peut pas lui *demander* une arête. Or une arête d'enveloppe
> manquante est manquante précisément parce qu'elle n'est pas de Delaunay.
> La retriangulation de cavité traite donc les **facettes**, pas les arêtes ;
> une arête bloquée reste l'affaire des bascules, et si elles n'y suffisent
> pas, du mode `allow_surface_nodes`.

### 4. Séparation matière / vide

Les facettes de l'enveloppe deviennent des **murs**. Deux tétraèdres séparés
par un mur sont de part et d'autre de la surface ; toute autre paire de
voisins est du même côté. L'intérieur s'obtient donc par inondation depuis
des cellules connues intérieures — et « connue intérieure » découle de
l'orientation que vous avez fournie, la matière étant du côté opposé à la
normale.

L'inondation est menée **des deux côtés**, et les deux résultats doivent
partitionner le maillage : chaque cellule intérieure ou extérieure, aucune
les deux, aucune ni l'une ni l'autre. Une cellule qui échappe à cela est une
fuite, signalée et non maillée en silence.

### 5. Qualité : raffinement puis chasse aux slivers

Un maillage *valide* n'est pas un maillage *utilisable*. Une tétraédrisation
des seuls nœuds d'une surface contient toujours des cellules dont les quatre
coins sont presque coplanaires, et une poignée suffit à rendre une matrice
élémentaire singulière.

**Raffinement de Delaunay.** On prend la cellule la plus mal formée et on
pose un nœud au centre de sa sphère circonscrite. Deux critères décident :

- le **rapport rayon-arête** \\( \rho = R/\ell \\), rayon circonscrit sur
  arête la plus courte. Un tétraèdre régulier vaut \\( \rho = \sqrt6/4
  \approx 0{,}61 \\) ; une aiguille ou un coin en donne un grand. Découper
  au-dessus d'un seuil \\( B \\) **termine pour tout \\( B > 2 \\)** — d'où
  le seuil juste au-dessus de 2 ;
- la **taille**, \\( R \\) comparé à `size`. Pour un tétraèdre régulier
  d'arête \\( a \\), \\( R = a\sqrt6/4 \\), ce qui convertit la longueur
  d'arête demandée en rayon visé.

Une règle est indispensable à la terminaison : **l'empiètement**. Un nœud
posé dans la sphère d'une facette du bord dégrade les cellules contre cette
facette au lieu de les améliorer, et le raffinement le redemande sans fin. La
réponse classique est de découper la facette ; l'enveloppe ne nous
appartenant pas, la cellule est simplement laissée telle quelle.

**C'est aussi ce qui borne `size` par le bas** : on ne peut pas mailler plus
fin que la surface ne le permet. Sur un cube à huit coins, tout centre
circonscrit empiète, et `size` n'a aucun effet ; il faut une enveloppe déjà
discrétisée à la finesse voulue.

**Chasse aux slivers.** Le raffinement ne peut rien contre le **sliver** :
quatre coins proches d'un même plan, répartis régulièrement sur un cercle.
Son arête la plus courte est honorable et sa sphère circonscrite petite —
\\( \rho \\) ne voit rien d'anormal — alors que son volume est presque nul.
C'est un théorème, pas une lacune d'implémentation : aucun nœud inséré ne le
casse. Le maillage est donc *amélioré* plutôt que subdivisé, par les deux
seuls mouvements qui ne changent pas ce qu'il remplit :

- **reconnexion** : les mêmes nœuds, joints autrement ;
- **retrait d'arête** : les arêtes du sliver ôtées, une à une ;
- **relaxation** : les mêmes liens, un nœud déplacé — les nœuds **intérieurs**
  seulement, jamais les vôtres.

Les trois sont jugés sur le plus petit **angle dièdre** des cellules touchées
et appliqués seulement s'ils l'améliorent, ce qui rend la passe monotone.

Le deuxième est celui qui tue réellement les slivers, et la raison est
géométrique. Un sliver est plat **en travers** d'une arête : une bascule 2-3
sur une de ses faces n'a donc souvent nulle part où aller — la paire de
cellules n'y est pas convexe — tandis que vider l'anneau de cellules autour de
l'arête fautive et le remplir autrement a toujours la place de le faire. Le
retrait d'arête généralise la bascule 3-2 et tout ce qui vient après.

Mesuré sur l'enveloppe de `tutoriel/maillage.py` (3 304 facettes), remplie à
quatre tailles, sans puis avec cette passe. Le plus petit angle dièdre médian
reste entre 44° et 47° dans tous les cas ; la dernière colonne est la forme
\\( \eta \\) de la cellule la plus plate, 1 pour un tétraèdre régulier et 0
pour un plat :

| `size` | mailles | part sous 10° | part sous 1° | \\( \eta \\) de la plus plate |
|---|---|---|---|---|
| 0,01 | 12 850 | 0,36 % → **0,00 %** | 0 → 0 | \\( 9{,}3\cdot10^{-2} \\) → \\( 3{,}5\cdot10^{-1} \\) |
| 0,005 | 97 574 | 0,89 % → **0,005 %** | 0,002 % → **0** | \\( 1{,}6\cdot10^{-2} \\) → \\( 7{,}9\cdot10^{-2} \\) |
| 0,0033 | 340 329 | 1,99 % → **0,10 %** | 0,017 % → **0,0003 %** | \\( 8{,}3\cdot10^{-3} \\) → \\( 1{,}9\cdot10^{-2} \\) |
| 0,0025 | 378 190 | 0,88 % → **0,018 %** | 0,0013 % → **0,0003 %** | \\( 9{,}3\cdot10^{-3} \\) → \\( 2{,}7\cdot10^{-2} \\) |

La colonne sous 1° est l'enjeu : une cellule sous 1° d'angle dièdre donne une
matrice élémentaire quasi singulière, et il n'en faut pas beaucoup pour
couler un calcul. La passe coûte de **1,1× à 1,7×** le temps de maillage,
selon la taille.

> **Ce que cela a demandé.** Une passe qui améliore un maillage doit être
> moins chère que sa construction, ce qui interdit deux réflexes. On ne
> **teste pas en faisant puis défaisant** : copier le maillage à chaque
> candidat coûte \\( O(n) \\) pour un échange qui coûte \\( O(1) \\), donc une
> passe qui fait \\( O(n) \\) échanges devient quadratique — mesuré, cela
> portait le maillage d'un million de mailles à 325 s au lieu de 77 s. Le
> remplacement de région décide donc **avant de muter** si les nouvelles
> cellules pavent bien la région, à partir de leurs seules faces. Et on ne
> paie pas la **recherche exhaustive** de remplissage pour une arête qu'on
> peut aussi bien laisser en place : la récupération du bord a besoin de
> complétude, une passe de qualité n'a besoin que d'une réponse.

## Quand l'enveloppe ne convient pas

Deux refus vous sont rendus, tous deux avec un lieu et une action.

**« cannot fit the envelope's edge/facet … »** — la surface ne peut pas être
retrouvée dans la triangulation. Soit elle n'admet réellement aucun maillage
sur ses nœuds, soit la récupération n'y arrive pas.

**« the mesh has N flat cell(s) that cannot be improved … »** — un sliver
dont les quatre coins sont des nœuds de l'enveloppe. Rien à insérer pour le
casser, rien à bouger puisque ses coins sont les vôtres, et le retrait
d'arête n'y arrive pas non plus. C'est devenu rare : aucune des enveloppes de
la suite de tests ne l'obtient plus. La mesure est
\\( \eta = 12\\,(3V)^{2/3} / \sum \ell^2 \\), qui vaut 1 pour un tétraèdre
régulier et 0 pour un plat ; le seuil de \\( 10^{-4} \\) est calibré et non
choisi — les cellules saines restent au-dessus de \\( 4\cdot10^{-2} \\), une
cellule réellement plate tombe sous \\( 10^{-7} \\).

Dans les deux cas, la réponse est de **rediscrétiser la surface** à cet
endroit — ou de laisser le mailleur le faire.

## `allow_surface_nodes` : ce qu'on échange

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:surface_nodes}}
```

Autorise le mailleur à **couper l'enveloppe plus fin** là où il ne peut ni la
retrouver ni la rendre utilisable.

Ce qui est conservé : la **forme**. Chaque nœud ajouté est posé sur l'arête
ou la facette qu'il divise, donc la surface reste la même surface, seule sa
triangulation s'affine. Ce qui est perdu : la **discrétisation** — la peau du
résultat ne coïncide plus avec le maillage surfacique fourni. Cela compte si
deux solides doivent partager une interface conforme ; cela ne compte pas si
l'enveloppe ne servait qu'à décrire une forme. Un avertissement sur `stderr`
indique combien de nœuds ont été ajoutés — et prévient que le maillage rendu
porte de ce fait un sous-maillage de plus.

**Et surtout, le résultat vous dit lesquels.** Un message sur `stderr` n'est
pas quelque chose sur quoi un script peut agir ; quand des nœuds ont été
posés sur l'enveloppe, le maillage rendu porte un **second sous-maillage de
`POI1`** qui les nomme. Il n'apparaît que dans ce cas — un maillage obtenu
sans rien ajouter n'a qu'un sous-maillage `TET4` :

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:surface_nodes_compte}}
```

Ce sous-maillage se visualise, se soustrait, sert de support de champ comme
n'importe quel autre. Attention en revanche si vous enchaînez : les opérateurs
qui attendent un maillage volumique pur veulent le sous-maillage `TET4` seul.

C'est aussi ce qui débloque. Une arête qu'on n'arrive pas à faire rentrer n'a
autrement aucune issue ; la couper en deux scinde le problème en deux plus
faciles, et une arête suffisamment entourée de ses propres subdivisions est
récupérée par le Delaunay tout seul. Ce qui a été ajouté est ensuite **rendu**
partout où le maillage veut bien s'en séparer : sur la plaque de
`tutoriel/maillage.py`, 13 nœuds posés, 10 repris, 3 restants.

## Pièges

**Orientation après `extrude`.** `extrude` ne vérifie pas que sa direction
est du côté de la normale de la surface source. Un `skin` de son résultat
revient donc souvent avec les normales **rentrantes**, et `triangulate_volume` le
refuse en vous renvoyant vers `invert`. C'est le cas du pipeline
`triangulate_surface` → `extrude` → `skin` ci-dessus.

**Uniquement du TRI3.** Un quadrangle n'a pas de plan unique à respecter ; il
est refusé plutôt que découpé en silence. Passez par `convert(peau, "TRI3")`.

**Nœuds confondus.** Deux nœuds distincts au même endroit déchirent la
surface ; utilisez `merge_nodes` au préalable. Le mailleur le signale
explicitement.

## Coût

Sur la plaque percée de `tutoriel/maillage.py`, maillée, extrudée et pelée à
la même taille que le remplissage :

| `size` | facettes | tétraèdres | temps |
| --- | --- | --- | --- |
| 0,01 | 3 304 | 12 850 | 2,3 s |
| 0,006 | 8 096 | 58 669 | 6,6 s |
| 0,004 | 17 824 | 195 919 | 19 s |
| 0,003 | 30 616 | 462 447 | 51 s |

Le coût est essentiellement **linéaire** en nombre de mailles produites.
L'opération est interruptible : `Ctrl+C` pendant un long maillage lève
`KeyboardInterrupt` sans rien laisser derrière.

Côté Rust, `ops::mesh::triangulate_volume(envelope, size, allow_surface_nodes)`,
et `triangulate_volume_cancellable(…, cancel)` pour la forme interruptible.
