# Formation débutant

Adaptation, à pyrucast, du plan d'une formation « Débuter avec Cast3M »
(présentation du logiciel, langage de commande, maillage, calcul thermique,
calcul mécanique, compléments) — le même déroulé pédagogique, mais avec
l'API Python de pyrucast et un unique fil rouge : **une plaque 
percée d'un trou**, encastrée d'un côté, chargée de l'autre.

Chaque section ci-dessous condense la construction pas à pas
en un script complet, testé, rangé dans le dossier
[`formation/`](https://github.com/Pyrucast/pyrucast/tree/master/formation)
du dépôt. Le code affiché dans le livre est **inclus directement depuis ces
fichiers** (pas de copie manuelle) : ce que vous voyez est ce qui s'exécute.

## Sommaire

1. [Présentation de pyrucast](presentation.md) — ce qu'est pyrucast, ce
   qu'il fait, comment l'installer.
2. [Python & conventions pyrucast](langage-python.md) — l'équivalent du
   chapitre « langage Gibiane » : objets, opérateurs, conventions de nommage.
3. [Maillage](maillage.md) — mailleur non structuré (triangulation avec
   trou) et mailleur structuré (balayage).
4. [Calcul thermique](thermique.md) — conduction, flux imposé, convection,
   source volumique, et le repérage géométrique des régions chargées.
5. [Calcul mécanique](mecanique.md) — élasticité linéaire, dilatation
   thermique, plasticité parfaite (pas à pas), contact unilatéral.
6. [Compléments](complements.md) — éléments structuraux, export de
   résultats, pour aller plus loin.

> **Portée.** pyrucast ne couvre aujourd'hui que la thermique (conduction,
> convection) et la mécanique des structures (élasticité, plasticité
> parfaite, endommagement, contact) — pas de fluides, de magnétostatique ni
> d'optimisation topologique. Ce chapitre s'y tient : les rubriques du
> support Cast3M qui n'ont pas d'équivalent testé dans pyrucast sont signalées
> en encadré, comme celui-ci, plutôt que passées sous silence.

Après la [compilation](../installation.md) de pyrucast :
```bash
pip install maturin
maturin develop --release --features extension-module,viz-interactive
```
ou son installé à partir de pypi
```bash
pip install pyrucast
```
Chaque script se lance directement depuis la racine du dépôt :

```bash
python formation/maillage.py
```
