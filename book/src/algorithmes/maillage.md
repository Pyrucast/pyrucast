# Construire et transformer un maillage

Les **mailleurs** (`ops::mesh`) construisent et transforment des
[maillages](../mesh.md). Chacun prend ses conteneurs par référence et renvoie
un **nouveau** `Mesh`. Côté Python ils sont exposés à plat
(`pyrucast.mesh.line`, …). Les pages qui suivent racontent comment travaille
chaque famille ; ce que prend et rend chaque fonction est dans la
[référence](../reference/mesh.md).

`barycenter` donne un nœud **neuf** au centre de chaque cellule, même quand deux
cellules partagent leurs nœuds. Les contraintes s'en servent pour créer leurs
nœuds-multiplicateurs, un par relation (cf. [Dirichlet](../contraintes/dirichlet.md)).

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-barycenter}}
```

**En quadratique, le maillage garde son degré.** Une ligne `SEG3` s'extrude en
`QUA8`, une surface `QUA8` en `HEX20`, une `TRI6` en `PENTA15`. Les nœuds
milieux de la source sont recopiés à chaque couche, et chaque arête dans le
sens de l'extrusion reçoit le sien à mi-hauteur. Le bord d'une surface
quadratique est fait de `SEG3`, et l'espace éléments finis d'un maillage
quadratique prend de lui-même l'interpolation `LAGRANGE2` (cf.
[Espace éléments finis](../fe-space.md)) :

```python
{{#include ../../../tests/python/test_doc_ops_maillage.py:mesh-extrude}}
```

**Les mailles extrudées sont directes.** Quand l'extrusion remplit l'espace —
une ligne en 2D, une surface en 3D —, chaque maille produite a un jacobien
positif, de quelque côté qu'on extrude : une maille source qui donnerait une
maille à l'envers (un segment extrudé vers sa droite, une facette extrudée
contre sa normale) est d'abord retournée. La peau d'un volume extrudé a donc
ses normales vers l'extérieur. Une ligne extrudée en 3D donne une surface, qui
n'a pas de sens absolu : ses mailles gardent l'ordre de la source.
