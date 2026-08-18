# Sauvegarde et relecture

Écrire des objets dans un fichier, et les relire **en gardant ce qu'ils partageaient**. Un dictionnaire à l'aller, un dictionnaire au retour.

```python
{{#include ../../tests/python/test_doc_sauvegarde_evolution.py:save_load}}
```

Les clefs sont libres : un espace, un accent, une unité — tout ce qu'une chaîne Python peut porter. C'est la raison du dictionnaire explicite plutôt que des mots-clefs.

## La garantie : le partage survit

C'est la propriété pour laquelle ce format existe. Deux champs bâtis sur un même support sont relus sur **un** support, pas sur deux copies aux mêmes nœuds :

```python
{{#include ../../tests/python/test_doc_sauvegarde_evolution.py:partage}}
```

L'union fusionne les zones qui partagent un support et laisse côte à côte celles qui n'en partagent pas — c'est l'observable la plus directe du partage. Sans cette garantie, un champ et son maillage relus ne combineraient plus : ils porteraient les mêmes nœuds sans être *le même* support, et toute l'arithmétique de champs tomberait à côté.

Le mécanisme tient en une phrase : les objets sont écrits sous des **identifiants locaux au fichier**, et un objet référencé deux fois n'est écrit qu'une fois. Une adresse mémoire n'aurait aucun sens dans un autre processus ; un numéro de fichier, si.

## On donne les racines, pas la liste des dépendances

`save` prend ce qui vous intéresse. Ce dont ces objets ont besoin suit tout seul :

```python
{{#include ../../tests/python/test_doc_sauvegarde_evolution.py:dependances}}
```

Il n'y a rien à énumérer, et rien à oublier.

## Relire ajoute, ne remplace pas

`load` fabrique des objets **neufs** et vous rend le dictionnaire. Ce qui vivait déjà dans votre session n'est pas touché : on peut relire deux fois le même fichier et obtenir deux graphes indépendants, ou relire un maillage à côté de celui qu'on manipule.

## Ce que le fichier ne porte pas

La règle est simple : **ce qui se recalcule ne s'écrit pas**.

Sortent donc du fichier tous les caches et toutes les mémoïsations — la matrice assemblée, la factorisation du solveur, le coloriage des mailles, les tables d'index paresseuses, la copie qu'un champ garde de la connectivité de son support. Tout cela se rebâtit à la première demande, exactement comme sur un graphe construit à la main :

```python
{{#include ../../tests/python/test_doc_sauvegarde_evolution.py:reassemblage}}
```

Ce n'est pas une économie de place accessoire : la copie qu'un champ nodal garde de sa connectivité pèse autant que le maillage lui-même, et un fichier qui la porterait la porterait une fois par champ.

### Les compteurs de références

Ils ne sont pas écrits non plus, et c'est délibéré : un fichier contient *certains* des objets qui référencent un nœud, pas tous. Un compteur sauvé décrirait un monde qui n'existe plus.

À la relecture, tout est recompté depuis zéro. Les objets se comptent seuls — chaque référence rendue par `load` compte pour une. Les nœuds sont réincrémentés par les sous-maillages relus.

**Conséquence à connaître** : un nœud relu n'est protégé que par les objets présents dans le fichier. Un `Node` que vous teniez dans une variable Python n'est pas archivé — c'est un atome de votre script, pas un objet du graphe. Sauver une `Coords` seule et la relire donne donc des nœuds à compteur nul, qu'un `gc()` collectera :

```python
{{#include ../../tests/python/test_doc_sauvegarde_evolution.py:refcount}}
```

C'est exactement l'état où l'on serait après avoir reconstruit les mêmes objets à la main sans en garder de `Node`. Pour qu'un nœud survive, sauvez ce qui l'utilise.

## Les valeurs simples

À côté des objets, le fichier accepte un `bool`, un `int`, un `float`, une `str`, et les listes **homogènes** de ces quatre types. De quoi ranger le pas de temps, le nom du cas de charge ou la liste des instants avec les champs auxquels ils se rapportent.

Trois refus, nommés plutôt que silencieux : une liste imbriquée ou un dictionnaire (hors périmètre), une liste hétérogène, et un entier hors des 64 bits du format — les entiers Python sont non bornés, le fichier ne l'est pas.

## Le fichier

```text
b"PYRUCAST"                       signature, 8 octets
version de format (u32)           toute autre valeur est refusée, jamais convertie
version du crate                  informative : elle sert au diagnostic, jamais au test
enregistrements                   (identifiant, type, octets), en ordre de dépendance
racines                           les clefs que vous avez données
```

Sauver deux fois les mêmes objets produit **le même fichier, octet pour octet** : les clefs sont triées, donc les identifiants sont distribués dans un ordre déterministe. Un fichier d'archive se compare, se met sous gestion de version, se hache.

Le format binaire est identique sous Linux et Windows : entiers petit-boutistes normalisés, `usize` sur 64 bits, `f64` IEEE-754, aucun chemin ni séparateur dépendant du système dans les données.

**Avant la version 1.0.0, le format peut changer sans préavis.** Une version inconnue est refusée avec un message qui nomme les deux numéros — jamais décodée à moitié.

## Ce n'est pas un format d'échange

Un fichier `.pyr` est le format de **session** de pyrucast : il sert à reprendre un calcul, pas à le publier. Pour donner des résultats à un autre outil, [`export_vtk`](operateurs/champs.md) existe pour ça, et ParaView le lit nativement.

La distinction n'est pas de la pudeur. Un format d'échange comme HDF5 apporte l'interopérabilité, la lecture partielle et la compression — mais **aucune notion d'identité d'objet ni de référence partagée**. La garantie du haut de cette page devrait y être reconstruite à l'identique, par-dessus. À l'inverse, un format de session n'a pas à être lisible par des tiers, et gagne à pouvoir casser tant que la bibliothèque n'est pas figée. Confondre les deux les abîme tous les deux.

## Côté Rust

```rust,ignore
use pyrucast::archive;

archive::save("etude.pyr", &[
    ("maillage fin", &mesh),
    ("T (°C)",       &temperature),
    ("pas de temps", &0.05_f64),
])?;

let mut objets = archive::load("etude.pyr")?;
let mesh2 = objets.mesh("maillage fin")?;      // erreur nommant clef, type attendu, type trouvé
let dt    = objets.float("pas de temps")?;
```

À l'écriture les types sont connus du compilateur, une tranche de paires suffit. À la relecture ils ne le sont pas : `load` rend une table nommée, dont on tire chaque objet avec son type attendu.

Le détail du mécanisme — la découverte des dépendances par la sérialisation elle-même, la détection de cycle, le crochet d'après-relecture — est dans [Modèle mémoire](memory-model.md) et dans la documentation du module `archive`.
