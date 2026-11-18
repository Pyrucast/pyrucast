# Sauvegarde et relecture

Écrire des objets dans un fichier, et les relire **en gardant ce qu'ils partageaient**. Un dictionnaire à l'aller, un dictionnaire au retour.

```python
{{#include ../../tests/python/test_doc_sauvegarde.py:save_load}}
```

Les clefs sont libres : un espace, un accent, une unité — tout ce qu'une chaîne Python peut porter. C'est la raison du dictionnaire explicite plutôt que des mots-clefs.

`save` accepte tous les objets de pyrucast — `Coords`, maillages, espaces EF, champs, évolutions, modèles, matrices, et chacune de leurs zones — et des [valeurs simples](#les-valeurs-simples). Les atomes, `Node` en tête, sont refusés par un `TypeError` : un nœud se sauve avec le maillage ou le `Coords` qui le porte.

## La garantie : le partage survit

C'est la propriété pour laquelle ce format existe. Deux champs bâtis sur un même support sont relus sur **un** support, pas sur deux copies aux mêmes nœuds :

```python
{{#include ../../tests/python/test_doc_sauvegarde.py:partage}}
```

L'union fusionne les zones qui partagent un support et laisse côte à côte celles qui n'en partagent pas : c'est la manière la plus directe de voir le partage. Sans cette garantie, deux champs relus porteraient les mêmes nœuds sans être posés sur *le même* support, et les opérations entre champs, qui apparient les zones par support, ne les apparieraient plus.

Le mécanisme tient en une phrase : les objets sont écrits sous des **identifiants propres au fichier**, et un objet référencé deux fois n'est écrit qu'une fois. Une adresse mémoire n'aurait aucun sens dans un autre processus ; un numéro dans le fichier, si.

## On donne les racines, pas la liste des dépendances

`save` prend ce qui vous intéresse. Ce dont ces objets ont besoin suit tout seul :

```python
{{#include ../../tests/python/test_doc_sauvegarde.py:dependances}}
```

Il n'y a rien à énumérer, et rien à oublier.

## Relire ajoute, ne remplace pas

`load` fabrique des objets **neufs** et vous rend le dictionnaire. Ce qui vivait déjà dans votre session n'est pas touché : on peut relire deux fois le même fichier et obtenir deux graphes indépendants, ou relire un maillage à côté de celui qu'on manipule.

Les nœuds relus appartiennent donc au `Coords` relu. Un `Node` du script qui a écrit le fichier y est refusé (« node 1 belongs to another Coords ») ; on lit avec le nœud relu, que `acquire` rend à partir de son identifiant :

```python
{{#include ../../tests/python/test_doc_sauvegarde.py:noeud_relu}}
```

## Ce que le fichier ne porte pas

La règle est simple : **ce qui se recalcule ne s'écrit pas**.

Sortent donc du fichier les caches — la factorisation du solveur, le coloriage des mailles, les tables d'index, le compagnon POI1 qu'une zone de maillage garde pour ses champs — et les valeurs assemblées d'une matrice. Les caches se rebâtissent à la première demande, comme sur un graphe construit à la main. Deux d'entre eux demandent un geste après relecture.

### Une matrice relue se réassemble

Une matrice revient avec ses blocs, sans ses valeurs assemblées. Le plus direct est de sauver le modèle et les matériaux, et d'assembler après relecture :

```python
{{#include ../../tests/python/test_doc_sauvegarde.py:reassemblage}}
```

Une matrice sauvée telle quelle repasse par `assemble()` avant de servir ; sans cela, `solve` la refuse (« Matrix has no assembled values ») :

```python
{{#include ../../tests/python/test_doc_sauvegarde.py:matrice_relue}}
```

### Le compagnon POI1 d'une zone de maillage

Une zone de maillage garde en cache son compagnon POI1, la zone POI1 de ses nœuds que `mesh.to_poi1` rend : les champs posés sur un même maillage reçoivent ainsi le même support. Ce compagnon n'est pas écrit : sur le maillage relu, `to_poi1` en fabrique un nouveau, distinct du support des champs relus avec lui. Les opérations entre champs ne les apparient alors pas : `|` en garde deux zones, aux mêmes nœuds. `restrict(champ, maillage)` porte un champ relu sur le compagnon du maillage relu :

```python
{{#include ../../tests/python/test_doc_sauvegarde.py:compagnon}}
```

### Les compteurs de références

Ils ne sont pas écrits non plus, et c'est délibéré : un fichier contient *certains* des objets qui référencent un nœud, pas tous. Un compteur sauvé compterait des objets que le fichier ne contient pas.

À la relecture, tout est recompté depuis zéro. Les objets se comptent seuls — chaque référence rendue par `load` compte pour une. Les nœuds sont recomptés par les zones de maillage relues.

**Conséquence à connaître** : un nœud relu n'est compté que par les objets présents dans le fichier. Un `Node` que vous teniez dans une variable Python n'est pas archivé. Sauver une `Coords` seule et la relire donne donc des nœuds à compteur nul : aucun objet du fichier ne les utilise.

```python
{{#include ../../tests/python/test_doc_sauvegarde.py:refcount}}
```

C'est exactement l'état où l'on serait après avoir reconstruit les mêmes objets à la main sans en garder de `Node`. Pour qu'un nœud reste compté après relecture, sauvez avec lui ce qui l'utilise.

## Les valeurs simples

À côté des objets, le fichier accepte un `bool`, un `int`, un `float`, une `str`, et les listes **homogènes** de ces quatre types. De quoi ranger le pas de temps, le nom du cas de charge ou la liste des instants avec les champs auxquels ils se rapportent.

Trois refus, chacun par un `TypeError` qui le nomme : une liste imbriquée ou un dictionnaire, une liste hétérogène, et un entier hors des 64 bits du format — les entiers Python sont non bornés, le fichier ne l'est pas.

## Le fichier

```text
b"PYRUCAST"                       signature, 8 octets
numéro de format (u32)            un autre numéro est refusé, jamais converti
version de pyrucast               pour information : la relecture ne la lit pas
enregistrements                   (identifiant, type, octets), chaque objet après ce dont il dépend
racines                           les clefs que vous avez données
```

Sauver deux fois les mêmes objets avec la même version de pyrucast produit **le même fichier, octet pour octet** : les clefs sont triées, donc les identifiants sont distribués dans un ordre déterministe. Un fichier d'archive se compare, se met sous gestion de version, se hache.

Le fichier est le même sous Linux et sous Windows : entiers rangés dans un ordre d'octets fixé (petit-boutiste), entiers de taille machine (`usize`) écrits par leur valeur sur une longueur variable, `f64` IEEE-754, aucun chemin ni séparateur propre au système dans les données. Un fichier écrit sur une machine 64 bits se relit sur une machine 32 bits, sauf une valeur trop grande pour elle, qui est refusée plutôt que tronquée.

**Le numéro de format change à chaque modification incompatible du fichier.** Avant la version 1.0.0 de pyrucast, aucune conversion n'est proposée : un fichier d'un autre numéro est refusé par un message qui nomme les deux numéros, jamais décodé à moitié.

## Ce n'est pas un format d'échange

Un fichier `.pyr` est le format de **session** de pyrucast : il sert à reprendre un calcul, pas à le donner à un autre logiciel. Pour cela, [`export_vtk`](visualization.md#export-vers-paraview-export_vtk) écrit un fichier que ParaView lit, et [`to_gmsh`](reference/export.md#export-to_gmsh) et [`to_medcoupling`](reference/export.md#export-to_medcoupling) passent maillages et champs à gmsh et à MED.

Les deux usages tirent en sens contraire. Un format d'échange décrit des maillages et des champs pour des logiciels qui ne connaissent pas pyrucast : le graphe de ses objets — modèles, matrices, évolutions, zones et supports partagés — n'y a pas sa place, et la garantie du haut de cette page s'y perdrait. Un format de session n'est lu que par pyrucast : il suit ses objets tels qu'ils sont, et son numéro de format change avec eux.

## Côté Rust

```rust,ignore
{{#include ../../tests/doc_conteneurs.rs:archive}}
```

À l'écriture les types sont connus du compilateur, une tranche de paires suffit. À la relecture ils ne le sont pas : `load` rend une table nommée, dont on tire chaque objet avec son type attendu.

Le détail du mécanisme — la découverte des dépendances par l'écriture elle-même, la détection d'un cycle, l'étape qui reconstruit à la relecture ce qui n'est pas écrit — est dans [Modèle mémoire](memory-model.md#sauvegarde--un-handle-devient-un-numéro) et dans la documentation du module `archive`.
