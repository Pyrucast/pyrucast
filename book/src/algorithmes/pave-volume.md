# Couche limite hexaédrique : `pave_volume`

{{#pyalgo mesh.pave_volume}}

`pave_volume(envelope, layers=1, thickness=None, size=None)` remplit la même
enveloppe fermée que `triangulate_volume`, mais met des **hexaèdres là où ils
comptent** — dans la couche contre le bord, où les gradients de contrainte et
de flux sont les plus raides et où la forme d'une maille décide de la
précision — et laisse l'intérieur, où le champ est lisse, aux tétraèdres.

```text
peau (QUA4 / TRI3)  ──►  décalage intérieur  ──►  HEX8 / PENTA6   (couche limite)
                                                        │
                     faces internes carrées ──► PYRA5   │           (le raccord)
                                                        ▼
                                 vide borné par des triangles seulement
                                                        │
                                          triangulate_volume  ──►  TET4
```

## Pourquoi les pyramides ne sont pas facultatives

Les faces internes de la couche sont **carrées**, et un tétraèdre n'en a
aucune. Découper chaque carré en deux triangles ne suffit pas : l'hexaèdre de
l'autre côté continue de voir une face carrée, avec un nœud suspendu en son
milieu. Le maillage n'est plus conforme et aucun solveur ne peut assembler à
travers cette face.

La pyramide est le seul élément qui présente un carré d'un côté et des
triangles de l'autre — c'est exactement ce que réclame le raccord. Ses
fonctions de forme se réduisent à celles d'un `QUA4` sur la base et restent
linéaires le long des arêtes vers le sommet, ce qui assure la continuité des
deux côtés : voir [PYRA5](../elements/pyra5.md).

Une fois chaque face carrée coiffée, ce qui reste du vide n'est plus borné que
par des triangles, et le mailleur tétraédrique existant prend le relais — en
mode **strict**, donc en réutilisant ces triangles tels quels : les deux
parties du maillage se rejoignent nœud à nœud.

## Un front, pas un décalage global

Décaler l'enveloppe entière d'une même distance est la version naïve, et elle
lâche dès que le solide cesse d'être épais partout : un seul étranglement
plafonne l'épaisseur de toutes les couches, sur toute la pièce. Le front fait
trois choses à la place.

**Il avance de ce que la place permet.** Le pas de chaque nœud est borné par la
distance du front à lui-même en ce point — la moitié de la distance à la
facette la plus proche à laquelle il n'appartient pas. Là où le solide est
épais la couche est pleine, là où il se pince elle s'amincit au lieu que tout
le maillage renonce. Demander une couche vingt fois plus épaisse que la pièce
ne produit donc plus une erreur mais une couche adaptée.

**Il place ses éléments localement.** Une facette qui ne peut pas avancer —
parce que la maille sortirait retournée — reste où elle est pendant que ses
voisines poursuivent. La marche ainsi créée est refermée par une **paroi
latérale**, un quadrangle neuf tendu entre l'ancienne arête et la nouvelle. Son
orientation n'est pas choisie mais forcée : si `A` avance et que sa voisine `B`
reste, l'arête `(u,w)` qu'elles partageaient n'est plus empruntée que par `B`,
dans le sens `(w,u)` ; il faut donc que quelque chose l'emprunte en `(u,w)`, et
emprunte la nouvelle arête en `(w',u')`. Le quadrangle `[u, w, w', u']` fait
exactement les deux, et le front reste une surface fermée et orientée.

**Il se coud.** Deux parties du front qui se retrouvent à distance de contact
sont soudées, ce qui referme une région mince au lieu d'y laisser un éclat
qu'aucune maille ne peut remplir. Deux critères, tous deux nécessaires : les
deux nœuds ne doivent pas partager de facette — les souder l'écraserait, ce
n'est pas une couture mais une dégénérescence — et leurs normales doivent
**se faire face**. Sans ce second critère, deux nœuds voisins d'une même nappe
lisse se soudent et replient la surface ; c'est ce qui rendait le cube
impossible à mailler pendant la mise au point.

**Il est lissé.** Les nœuds qu'il a créés sont relâchés sous garde de validité,
si bien qu'un pas raccourci par la place disponible ne reste pas en coude. La
garde est le jacobien normalisé lui-même, pas un indicateur qui lui
ressemblerait, et le balayage est de Gauss–Seidel : chaque déplacement est jugé
sur le maillage tel qu'il est, donc la pire maille ne peut que s'améliorer.

## Décaler n'est pas « déplacer chaque nœud selon sa normale »

Moyenner les normales des facettes incidentes donne une direction, et cette
direction ne suffit pas. Déplacez le coin d'un cube de \\( t \\) le long de la
normale moyenne \\( (1,1,1)/\sqrt3 \\) et chacune des trois faces ne s'écarte
que de \\( t/\sqrt3 \\) : la couche est la plus mince là où la géométrie
tourne, c'est-à-dire là où elle peut le moins se le permettre. Pire, au coin
d'un tétraèdre la normale moyenne est **tangente** à l'une des faces
incidentes, qu'un déplacement le long d'elle ne décale donc pas du tout.
Aucun facteur d'échelle ne rattrape cela : c'est la direction qui est fausse.

Le nœud décalé doit être le point où les facettes incidentes, chacune poussée
de \\( t \\) vers l'intérieur, se rencontrent :

\\[
\mathbf{d} \cdot \mathbf{n}_j = -t \quad \text{pour chaque facette incidente } j,
\\]

soit trois équations à trois inconnues à un coin, davantage sur une surface
lisse, moins sur une arête. Résoudre au sens des **moindres carrés**, par les
équations normales \\( (N^{\\!\top}\\!N)\\,\mathbf{d} = -t\\,N^{\\!\top}\mathbf{1} \\),
couvre les trois cas d'un coup et rend l'intersection exacte quand elle
existe. Au coin du cube : \\( \mathbf{d} = -t(1,1,1) \\).

## Convention

`envelope` est une surface **fermée** de facettes `QUA4` et/ou `TRI3`, de
normales **sortantes de la matière** — même convention que
`triangulate_volume`, si bien que la peau d'un maillage (`skin`) s'y branche
directement. Ses nœuds sont réutilisés tels quels. Une enveloppe ouverte, ou
dont les facettes se contredisent sur l'orientation, est refusée en nommant
l'arête fautive.

`layers` couches sont poussées vers l'intérieur, chacune de `thickness` de
profondeur ; `thickness=None` prend la longueur d'arête moyenne de
l'enveloppe, ce qui donne des mailles à peu près cubiques. `size` est la
taille visée pour le cœur tétraédrique.

Le résultat porte un sous-maillage `HEX8` (issu des facettes carrées), un
`PENTA6` (des triangulaires), un `PYRA5` (le raccord) et un `TET4` (le cœur),
chacun présent seulement s'il n'est pas vide.

## Exemple Python

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-pave_volume}}
```

## Pièges

- **Orientation.** Une enveloppe retournée est refusée en le disant ;
  `pyrucast.mesh.invert` la remet à l'endroit.
- **Épaisseur.** Une couche plus épaisse que le solide ne peut pas rentrer :
  le décalage retourne l'enveloppe et l'opérateur sort en erreur en nommant la
  couche fautive.
- **Le cœur peut refuser.** Le mailleur tétraédrique travaille en mode strict,
  donc sans ajouter de nœud sur la surface intérieure ; s'il n'y arrive pas,
  l'erreur le dit et une couche plus mince, ou une enveloppe plus fine, est la
  réponse habituelle.

## Qualité et coût

Mesuré sur des cubes de 6³ à 16³ mailles de peau, en `--release`
(`cargo test --release -- --ignored volume_report --nocapture`) :

| cas | mailles | temps | débit | jacobien médian | inversées |
|---|---|---|---|---|---|
| cube 6³, 1 couche | 4 054 | 0,22 s | 18 700 /s | 0,374 | 0 |
| cube 8³, 1 couche | 7 422 | 0,38 s | 19 500 /s | 0,345 | 0 |
| cube 12³, 1 couche | 16 454 | 0,80 s | 20 600 /s | 0,418 | 0 |
| cube 16³, 1 couche | 35 015 | 1,72 s | 20 300 /s | 0,431 | 0 |

Le coût est **linéaire**, autour de 20 000 mailles/s et 50 µs par maille. Par
type, sur le cube 16³ :

| type | minimum | médiane |
|---|---|---|
| `HEX8` | 0,577 | **0,987** |
| `PYRA5` | 0,181 | 0,319 |
| `TET4` | 0,040 | 0,431 |

La couche hexaédrique est donc quasi parfaite — c'est le but — et le minimum de
0,577 n'est pas un défaut mais la valeur exacte d'un coin à 60°, celle des
hexaèdres qui suivent une arête du cube. Les pyramides sont les mailles les
plus médiocres du lot, ce qui est attendu d'un élément de raccord aplati.

Une seconde couche coûte cher : le cube 6³ tombe à 2 100 mailles/s, la
tétraédrisation du cœur devenant nettement plus difficile.

## Ce que l'enveloppe devient

Elle est **respectée à la maille près**. Ses nœuds sont réutilisés tels quels
(mêmes identifiants, mêmes positions), ils sont marqués immobiles donc le
lissage ne les touche pas, et chaque facette devient **exactement une face de
maille** — un `QUA4` la face d'un `HEX8`, un `TRI3` celle d'un `PENTA6`. Aucun
nœud n'est ajouté sur le bord. La couture protège explicitement ce contrat :
entre deux nœuds candidats, celui de l'enveloppe est toujours le survivant, et
si les deux en sont, la couture est refusée.

C'est aussi pour cela que le cœur tétraédrique tourne en mode **strict** : en
mode permissif il ajouterait des nœuds sur la surface intérieure et ne
rejoindrait plus les pyramides. Cette exigence est la contrepartie du contrat,
et la principale source d'échec.

## Limitations actuelles

- **Deux cas sur huit échouent**, tous deux au cœur : une plaque mince et un
  barreau, c'est-à-dire les géométries où le front se referme réellement sur
  lui-même. La couture soude bien, mais un repli *partiel* laisse un vide que
  le mailleur tétraédrique, en mode strict, ne sait pas remplir. C'est la
  limite principale aujourd'hui.
- La profondeur des pyramides est fixée au quart de l'arête de leur base. Plus
  profondes, elles seraient mieux formées mais finiraient par se traverser.
- **Les parois latérales ne servent jamais sur les cas mesurés.** Le mécanisme
  est là et testé — retenir une facette lève bien quatre parois et le front
  reste fermé — mais sur les huit géométries du banc, le plafonnement par la
  place suffit toujours à rendre les mailles valides, et aucune facette n'a
  besoin d'être retenue. C'est une capacité en réserve, pas un moteur en
  service.
- Le plastering **complet** — remplir tout le volume d'hexaèdres par front
  avançant, sans cœur tétraédrique — reste un problème ouvert. Sandia, qui l'a
  inventé, l'a abandonné : la fermeture du vide central bute sur des
  obstructions topologiques qu'une méthode locale ne voit pas. Le cœur
  tétraédrique n'est donc pas un raccourci mais l'état de l'art.
