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
