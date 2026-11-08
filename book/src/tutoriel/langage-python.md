# Python & conventions pyrucast

Un script pyrucast est un programme **Python ordinaire** : il n'y a pas de
langage à part à apprendre. Cette page rassemble ce qu'il faut savoir pour
lire et écrire les scripts du tutoriel, en deux parties :

1. **le Python des scripts** — chaque notion de Python, dans l'ordre où on la
   rencontre, suivie aussitôt de ce que pyrucast en fait : ses raccourcis ;
2. **les conventions de pyrucast** — où trouver une fonction, quand elle existe
   aussi en méthode, comment enchaîner les appels.

Inutile de tout retenir d'une traite : revenez ici quand une ligne d'un script
vous arrête. Tous les extraits de cette page sont exécutés à chaque
vérification du projet.

## Partie 1 — Le Python des scripts, et les raccourcis de pyrucast

### Un script, et comment le lancer

Un script est un fichier texte d'extension `.py`. Python le lit **de haut en
bas**, une instruction par ligne. Tout ce qui suit un `#` est un
**commentaire** : Python l'ignore, il est là pour le lecteur. Le texte entre
triples guillemets `"""…"""` en tête d'un script le présente.

On le lance depuis un terminal, dans le dossier du script :

```bash
python mon_script.py
```

On peut aussi taper les instructions une à une : la commande `python` seule
ouvre une session interactive (invite `>>>`), où chaque ligne s'exécute dès
qu'on appuie sur Entrée. Un *notebook* Jupyter fonctionne de la même façon,
cellule par cellule.

### Nommer des valeurs

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:valeurs}}
```

Le signe `=` **range une valeur sous un nom** : `COTE` vaut désormais `0.10`,
et on écrit `COTE` partout où l'on veut cette valeur. Un nombre décimal prend
un **point** ; `1.0e6` se lit « 1,0 × 10⁶ » ; les `_` d'un grand nombre ne
servent qu'à le lire. Les noms en majuscules sont une convention, pas une
règle : ils signalent les **données** du calcul, posées une fois en tête du
script.

Le texte s'écrit **entre guillemets**. pyrucast en reçoit beaucoup : un type
d'élément (`"QUA4"`), un nom de composante (`"T"`), un nom de coefficient
(`"k"`).

### Regrouper des valeurs : listes, tuples, dictionnaires

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:collections}}
```

- Une **liste**, entre crochets `[…]`, aligne des valeurs : les coordonnées
  d'un point, les composantes d'un vecteur, une suite d'instants.
- Un **tuple**, entre parenthèses `(…)`, fait de même pour un groupe qui ne
  change pas : une couleur `(rouge, vert, bleu)`, un point de vue.
- Une liste de **paires** `(nom, valeur)` est la forme sous laquelle pyrucast
  reçoit les coefficients d'un matériau.
- Les éléments se numérotent **à partir de 0** : `point[0]` est le premier.

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:dictionnaire}}
```

Un **dictionnaire**, entre accolades `{…}`, range des valeurs **par nom**.
C'est sous cette forme que le [calcul pas à pas](../pas-a-pas.md) reçoit ses
données.

### Importer pyrucast

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:importer}}
```

`import pyrucast as pc` rend tout pyrucast disponible sous le nom court `pc`.
Le point se lit « qui appartient à » : `pc.Coords` est la classe `Coords` du
paquet, `pc.mesh.line` la fonction `line` de son module `mesh`.

`import os` charge un module livré avec Python : les scripts du tutoriel s'en
servent pour lire une variable d'environnement et composer des chemins de
fichiers.

Un script peut aussi reprendre des noms d'un autre script du **même dossier** :
`tutoriel/thermique.py` commence par
`from maillage import HEIGHT, HOLE_RADIUS, LENGTH, OUT, THICKNESS, show, structured_mesh`,
et calcule ainsi sur le maillage construit par `tutoriel/maillage.py`, sans le
recopier. C'est pourquoi les deux fichiers doivent rester côte à côte.

### Appeler une fonction

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:arguments}}
```

Une fonction s'appelle avec ses **arguments** entre parenthèses :

- **par position**, dans l'ordre que la fonction attend — pour `extrude`, le
  maillage à balayer, le vecteur, le nombre de couches ;
- **par nom**, `angle_deg=45`, quand l'ordre ne suffit pas à se faire
  comprendre ou que l'argument est facultatif ;
- un argument facultatif omis prend sa **valeur par défaut** : sans
  `angle_deg`, `border` rend tout le bord d'un seul tenant, au lieu de le
  couper en quatre côtés.

Trois valeurs reviennent souvent : `True` (vrai), `False` (faux), et `None`,
qui veut dire « rien » — par exemple « pas de fichier, ouvre une fenêtre ».

Pour savoir ce qu'attend une fonction, on demande son aide dans une session
interactive :

```text
>>> help(pc.mesh.line)
line(a, b, n_elems, element_type=None)
    Build a line of `n_elems` elements from node `a` to node `b`.

    `element_type` is `"SEG2"` (default) or `"SEG3"`.
```

Les éditeurs de code (VS Code, PyCharm…) affichent la même aide, et
proposent les noms de fonctions et d'arguments au fil de la frappe.

### Les objets : méthodes et propriétés

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:objets}}
```

Un **objet** — un magasin de nœuds, un maillage, un champ — porte avec lui
deux sortes de choses, qu'on atteint par un point :

- des **méthodes**, qui agissent : elles s'appellent **avec des
  parenthèses**, `plaque.element_types()` ;
- des **propriétés**, qui sont des valeurs : **sans parenthèses**,
  `plaque.cell_count`. Les nombres que l'on compte (mailles, nœuds,
  composantes) sont des propriétés dans pyrucast.

Certaines propriétés se modifient : `face_color` fixe la couleur de tracé
d'une zone de maillage. Les scripts du tutoriel s'en servent pour colorier les
régions chargées.

### Les crochets : zones, composantes, valeurs

La plupart des objets pyrucast sont **découpés en zones** : un maillage a une
zone par type d'élément ou par morceau, un champ une zone par support, un
modèle une zone par terme. Les crochets, qui en Python désignent un élément
d'une liste, désignent ici une zone :

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:crochets}}
```

- `x[0]` est la première zone, `x[-1]` la dernière ; `len(x)` en donne le
  nombre.
- `x[1:]` (« de la deuxième à la fin ») ou `x[1:-1]` (« ni la première ni la
  dernière ») rendent un objet de même nature, réduit à ces zones.
- Une zone n'est pas une copie : la modifier, c'est modifier l'objet entier.
- Sur un champ, **un nom** entre crochets garde une seule composante :
  `temperature["T"]` laisse de côté `"lambda"`, les multiplicateurs de
  Lagrange que la résolution ajoute. Une liste de noms en garde plusieurs.
- Sur une zone de champ, `[nœud, composante]` lit ou écrit **une valeur**.

Le détail est dans [Champ](../field.md#lire-un-champ).

### Les opérateurs : réunir et calculer

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:operateurs}}
```

Sur des nombres, `+ - * / **` (la puissance) font ce qu'on attend. pyrucast
leur donne un sens sur ses objets :

- `|` **réunit** : deux nœuds en un maillage `POI1`, deux maillages en un
  maillage à plusieurs zones, deux modèles en un seul modèle, deux champs en
  un seul champ. Il juxtapose, il **ne somme jamais** (voir plus bas, *Réunir
  sans sommer*).
- `+ - * / **` **calculent sur les valeurs** d'un champ, avec un nombre ou
  avec un autre champ posé sur les mêmes nœuds : `temperature + 273.15`
  passe en kelvins, `u + du` met à jour un déplacement. `-champ` change le
  signe. Les règles, entre deux champs, sont dans
  [Champ](../field.md#calculer-avec-des-champs).
- `>=`, `>`, `<=`, `<` comparent un champ à un nombre et rendent un **masque** :
  un champ qui vaut 1 là où la comparaison est vraie, 0 ailleurs.
- Les matrices s'additionnent et se multiplient ou se divisent par un nombre,
  et `K * u` multiplie une matrice par un champ : c'est ainsi que s'écrit le
  pas en temps `C / dt + K` de la [conduction transitoire](../thermique.md).

### Répéter : `for`, et l'indentation

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:boucle}}
```

`for cote in cotes:` répète le bloc qui suit **pour chaque zone** de `cotes`,
en la rangeant à chaque tour sous le nom `cote`. Le bloc répété est celui qui
est **décalé vers la droite** (quatre espaces) : en Python, l'indentation fait
partie du langage. Un décalage oublié est une erreur, un décalage en trop
change le sens du programme.

`enumerate` donne en plus le numéro de chaque tour, `0, 1, 2…` ; `%` est le
reste de la division, qui fait ici tourner les couleurs de la palette — c'est
exactement ce que fait `tutoriel/maillage.py` pour colorier les faces d'une
peau. Pour répéter un bloc `n` fois, on écrit `for i in range(n):`, où `i`
vaut `0`, `1`, …, `n - 1`.

### Choisir : `if`

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:condition}}
```

`if` n'exécute son bloc (décalé, lui aussi) que si la condition est vraie ;
`else` donne l'autre branche. `None`, `0` et un texte vide comptent pour
« faux ». Les scripts du tutoriel l'emploient pour choisir entre une fenêtre et
un fichier : `os.environ.get` lit une *variable d'environnement* — un réglage
posé dans le terminal avant de lancer le script, ici le dossier où écrire les
figures — et rend `None` si elle n'existe pas.

### Définir une fonction : `def`

Quand un même geste revient souvent, on le range dans une **fonction**. Voici
celle que `tutoriel/maillage.py` définit pour tracer ses figures :

```python
{{#include ../../../tutoriel/maillage.py:figures}}
```

- `def show(…):` définit la fonction `show` ; son corps est le bloc décalé.
- `mesh`, `title`, `file`, `wireframe` sont ses **paramètres** ;
  `wireframe: bool = False` lui donne une valeur par défaut.
- `: pc.Mesh`, `: str`, `-> None` disent quels types sont attendus et rendus.
  Ce sont des indications pour le lecteur et l'éditeur : Python ne les vérifie
  pas.
- `x if condition else y` est un choix en une ligne : le chemin du fichier si
  `OUT` est défini, `None` (donc une fenêtre) sinon.
- Une fonction rend un résultat par `return`. `structured_mesh` en rend deux,
  `return grid, volume_structure`, que `tutoriel/thermique.py` reçoit par
  `_, volume = structured_mesh(plot=False)` — le nom `_` signale une valeur
  dont on ne se sert pas.

Chaque script range enfin son travail dans une fonction `main()`, et se termine
par :

```python
{{#include ../../../tutoriel/premier_calcul.py:lancement}}
```

Cette condition n'est vraie que si l'on **lance** le fichier. Quand un autre
script l'importe, `main()` ne s'exécute pas : `tutoriel/thermique.py` reprend
le maillage de `tutoriel/maillage.py` sans retracer toutes ses figures.

### Un bloc dans un contexte : `with`

`with` exécute un bloc décalé dans un **contexte**, et le défait à la sortie,
même si le bloc s'arrête sur une erreur. pyrucast s'en sert pour calculer
dans une autre configuration des coordonnées — la pièce déformée plutôt que la
référence :

```python
{{#include ../../../tests/python/test_doc_coords.py:in_config}}
```

C'est un usage avancé, dont le tutoriel ne se sert pas : voir
[Coordonnées](../coords.md#plusieurs-configurations).

### Afficher : `print`

`print(…)` écrit dans le terminal. Un texte précédé de `f`, une
*f-string*, y insère des valeurs entre accolades :
`print(f"T max = {temperature.max('T'):.2f} °C")` affiche la température
maximale avec deux chiffres après la virgule.

Tout objet pyrucast s'affiche à trois niveaux de détail :

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:listings}}
```

`print(x)` donne le résumé d'une ligne, `repr(x)` la structure, `x.dump()` le
contenu complet — voir
[Aspect informatique](../aspect-informatique.md#trois-niveaux-de-listing).

### Lire un message d'erreur

Quand une instruction échoue, Python s'arrête et affiche un message de ce
genre :

```text
Traceback (most recent call last):
  File "mon_script.py", line 14, in <module>
    bas = cotes.unit()
RuntimeError: expected a unitary Mesh (exactly one submesh(es)), found 4
```

Il se lit **par le bas**. La dernière ligne dit ce qui ne va pas — le type de
l'erreur, puis le message, en anglais. Les lignes au-dessus disent où :
le fichier, le numéro de ligne, l'instruction. Ici, `unit()` rend la zone
**unique** d'un maillage, et `cotes` en a quatre : il fallait choisir une zone,
`cotes[0]`. Les messages de pyrucast disent ce qui était attendu et ce qui a
été trouvé.

## Partie 2 — Les conventions de pyrucast

### Trois sortes d'objets

- **`Coords`**, le **magasin de nœuds**. Un script en crée un, en général, et
  tous les mailleurs y déposent leurs nœuds : c'est ce qui fait que deux
  maillages construits côte à côte partagent leurs nœuds communs.
- **`Node`**, un **nœud** de ce magasin.
- Les **conteneurs**, découpés en zones, chacun avec sa page dans la partie
  [Objets](../objets.md) :

| objet | contient | une zone |
|---|---|---|
| `Mesh` | un maillage | `SubMesh` |
| `FiniteElementSpace` | l'interpolation choisie sur un maillage | `SubFiniteElementSpace` |
| `NodeField` | des valeurs aux nœuds | `SubNodeField` |
| `ElementField` | des valeurs aux points de Gauss | `SubElementField` |
| `Model` | une physique, ses conditions, ses chargements | `SubModel` |
| `Matrix` | une matrice creuse | `SubMatrix` |
| `Evolution` | des valeurs tabulées (une courbe, un champ qui évolue) | `SubEvolution` |

Un objet se **déclare** à partir d'une description, et naît souvent vide :

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:declarer}}
```

Tout le reste se **calcule**, par des fonctions.

### Les fonctions, rangées par ce qu'elles produisent

Une fonction vit dans le module qui porte le nom de **ce qu'elle rend**. Pour
trouver une fonction, demandez-vous ce que vous voulez obtenir :

| module | rend | exemples du tutoriel |
|---|---|---|
| `pyrucast.mesh` | un maillage | `line`, `arc`, `circle`, `extrude`, `sweep`, `triangulate_surface`, `skin`, `border`, `select`, `elements_on`, `points_on_cylinder`, `to_poi1` |
| `pyrucast.node_field` | un champ aux nœuds | `positions`, `restrict`, `external_forces` |
| `pyrucast.element_field` | un champ aux points de Gauss | `material_field`, `gradient`, `integrate_behavior` |
| `pyrucast.model` | un modèle | `heat_conduction`, `boundary_transfer`, `flux`, `dirichlet`, `elasticity` |
| `pyrucast.matrix` | une matrice | `stiffness`, `mass` |

`pyrucast.mesh.select` rend un maillage — les nœuds retenus — bien qu'il
parte d'un champ : c'est le résultat qui décide du module.

Les fonctions qui ne rendent aucun de ces objets sont rangées par activité :

| module | activité | exemple |
|---|---|---|
| `pyrucast.solver` | résoudre un système | `solve` |
| `pyrucast.measure` | mesurer un nombre (une intégrale…) | `integral` |
| `pyrucast.export` | écrire pour un autre logiciel | `export_vtk` |
| `pyrucast.coords` | déplacer les nœuds du magasin | `displace` |
| `pyrucast.field` | les fonctions mathématiques, sur tout champ | `sqrt`, `cos`, `exp` |
| `pyrucast.stepping` | le calcul pas à pas | `step_by_step` |

Restent, au niveau du paquet lui-même, `pyrucast.save` et `pyrucast.load`, qui
écrivent et relisent une session (voir [Sauvegarde](../sauvegarde.md)).

### La même opération en méthode

Beaucoup de fonctions existent aussi en **méthode de leur premier argument** :

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:methode}}
```

Les deux écritures font exactement la même chose, et la méthode a la même
aide que la fonction. Une fonction a sa méthode quand trois conditions sont
réunies :

1. son premier argument est l'objet que l'opération **transforme** ;
2. elle **rend un objet** pyrucast ;
3. elle a un sens pour **n'importe quel** objet de ce type.

| fonction | méthode |
|---|---|
| `pyrucast.mesh.extrude(ligne, v, n)` | `ligne.extrude(v, n)` |
| `pyrucast.mesh.border(plaque)` | `plaque.border()` |
| `pyrucast.mesh.skin(volume)` | `volume.skin()` |
| `pyrucast.mesh.consolidate(m)` | `m.consolidate()` |
| `pyrucast.mesh.to_poi1(m)` | `m.to_poi1()` |
| `pyrucast.node_field.positions(m, ["X"])` | `m.positions(["X"])` |
| `pyrucast.mesh.select(champ, ge=…)` | `champ.select(ge=…)` |
| `pyrucast.node_field.restrict(champ, m)` | `champ.restrict(m)` |
| `pyrucast.element_field.material_field(modele, …)` | `modele.material_field(…)` |
| `pyrucast.node_field.external_forces(modele, mat)` | `modele.external_forces(mat)` |
| `pyrucast.matrix.stiffness(modele, mat)` | `modele.stiffness_matrix(mat)` |
| `pyrucast.solver.solve(K, f)` | `K.solve(f)` |

Quand le module précisait la nature du résultat, la méthode la reprend dans
son nom : `pyrucast.matrix.stiffness` devient `stiffness_matrix`.

Restent seulement des fonctions :

- les physiques de `pyrucast.model` : leur premier argument est l'espace sur
  lequel on **déclare** la physique, pas un objet qu'elles transforment ;
- `pyrucast.mesh.line(a, b, n)`, dont le premier argument est un nœud ;
- `pyrucast.element_field.deformation`, qui n'a de sens que pour un champ
  portant des déplacements `u_x`, `u_y`… — pas pour n'importe quel champ.

### Composer les appels : imbriquer ou enchaîner

Le résultat d'un appel peut servir directement d'argument au suivant. Avec les
fonctions, les appels **s'imbriquent**, et se lisent de l'intérieur vers
l'extérieur :

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:maillage_imbrique}}
```

Avec les méthodes, ils **s'enchaînent**, et se lisent de gauche à droite, dans
l'ordre des opérations — « une ligne, balayée ; son bord, en nœuds » :

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:maillage_enchaine}}
```

La résolution du premier calcul, des deux façons :

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:resolution_imbrique}}
```

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:resolution_enchaine}}
```

Et deux lignes de `tutoriel/thermique.py`, écrites en méthodes (la forme
fonction est rappelée en commentaire) :

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:chaines_thermique}}
```

Les scripts du tutoriel écrivent tout en fonctions, une opération par ligne :
c'est la forme que décrit la documentation de chaque opérateur. Les deux
formes se mélangent librement ; choisissez celle qui se lit le mieux.

### Réunir sans sommer : `|` et les zones

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:composer}}
```

`|` ajoute chaque terme au modèle comme une **zone** de plus : `len(modele)`
en compte trois. `unit()` rend la zone unique d'un objet qui n'en a qu'une — et
s'arrête sur une erreur sinon, comme plus haut. `pyrucast.mesh.consolidate`
fond en une seule les zones d'un maillage qui ont le même type d'élément ;
plusieurs mailleurs l'exigent.

Sur les champs, `|` juxtapose aussi les zones : à un nœud que deux zones
partagent, il **ne somme pas** les valeurs. Pour additionner deux chargements
qui se touchent, on les ramène d'abord sur les mêmes nœuds avec
`pyrucast.node_field.restrict`, puis on les somme par `+` — c'est l'objet du
passage *Trois chargements qui se touchent* de la page
[Calcul thermique](thermique.md).

### Les noms des composantes

```python
{{#include ../../../tests/python/test_doc_tutoriel.py:composantes}}
```

Chaque physique déclare ses inconnues par **paires** : une variable
**primale**, l'inconnue elle-même, et sa **duale**, la grandeur qui lui
répond — `T` et le flux `q` en thermique, `u_x` et la force `f_x` en
mécanique. `dirichlet` reçoit le nom de la **primale**, celle que l'on impose,
et trouve la duale dans le modèle visé ; `dual_of` la donne.

Les coefficients des matériaux se désignent eux aussi par leur nom : « k »
pour la conduction, « phi_q » pour une densité de flux, « h_T » et « a_ext_T »
pour la convection. `pyrucast.element_field.material_field` sait lesquels le
modèle réclame. Chaque page de [Détails des physiques](../physiques.md) donne
la liste des siens.

Suite : [Maillage](maillage.md).
