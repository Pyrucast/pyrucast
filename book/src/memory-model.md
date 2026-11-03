# Modèle mémoire

En Python, la mémoire de pyrucast se gère seule : on ne libère rien, on ne
verrouille rien. Quatre comportements sont à connaître :

- **Un objet est partagé, pas copié.** Une affectation ou une zone lue par
  `m[i]` désignent le même objet, une union `|` reprend les zones sans les
  copier. Un opérateur (`+`, `restrict`, `solve`…) rend un objet neuf.
- **Un objet vit tant que quelque chose le tient** : une variable Python ou un
  autre objet pyrucast. Un champ tient son support, un maillage ses zones, une
  zone son `Coords`. Ce que plus rien ne tient est libéré aussitôt.
- **Un nœud reste dans son `Coords`** tant que le `Coords` existe.
- **Certains résultats sont gardés en cache** par l'objet dont ils sont tirés,
  et resservent aux calculs suivants.

## Partager plutôt que copier

```python
{{#include ../../tests/python/test_doc_memoire.py:partage}}
```

`print` d'un agrégat affiche le repère de chaque zone (`#8baac0`) : `gauche` et
`tout` montrent le même, ils partagent cette zone. `U = T` ne copie rien :
écrire par `U` modifie `T`. `T + T` rend un champ neuf, qu'on modifie sans
toucher `T`.

Une zone partagée ne change pas sous les objets qui l'indexent : dès qu'un
espace éléments finis, un champ ou une matrice s'y pose, elle est **scellée** et
sa connectivité ne bouge plus (voir [Scellement](mesh.md#scellement)). Pour repartir d'une zone
sans la partager, on la copie avec `mesh.duplicate()` ou
`pyrucast.mesh.copy`.

## Durée de vie

```python
{{#include ../../tests/python/test_doc_memoire.py:duree_de_vie}}
```

Après le `del`, plus rien ne tient les zones SEG2 : elles sont libérées
aussitôt et rendent leurs nœuds. Le champ `T` tient son support, le nuage des nœuds
de `gauche`, qui reste.

`coords.refcount(id)` compte ce qui utilise un nœud : les `Node` tenus et chaque
maille qui le contient (voir
[Coordonnées](coords.md#identité-et-durée-de-vie-des-nœuds)). Un nœud n'est
jamais retiré de son `Coords` : sa place et ses positions y restent même quand
plus rien ne l'utilise, et leur mémoire n'est rendue qu'avec le `Coords`.

## Ce qui est mis en cache

Un cache est tiré d'un objet et gardé par lui. Il se calcule à la première
demande, resert tant que l'objet ne change pas, et disparaît avec lui. Aucun
n'est écrit dans une sauvegarde : après relecture, chacun se recalcule à la
première demande. En Python, un cache ne se voit qu'au temps de calcul : le
premier appel le construit, les suivants s'en servent.

| Gardé par | Cache | Utilité |
|---|---|---|
| une zone de maillage | son nuage de nœuds (une zone POI1) | les champs posés sur un même maillage reçoivent le même support et se combinent nœud à nœud — [Maillage](mesh.md#ce-quune-zone-garde-en-cache) |
| une zone de maillage | la table « nœud → ligne » | lire ou écrire la valeur d'un nœud sans le chercher — [Maillage](mesh.md#ce-quune-zone-garde-en-cache) |
| un sous-espace éléments finis | la coloration des mailles | assembler et accumuler aux nœuds en parallèle, sans conflit d'écriture — [Espace éléments finis](fe-space.md#ce-qui-est-mis-en-cache) |
| un modèle | le motif de chaque genre de matrice | réassembler (une itération de Newton) sans reconstruire la structure de la matrice — [Modèle](model.md#ce-qui-est-mis-en-cache) |
| une matrice | sa factorisation | résoudre d'autres seconds membres sans refactoriser — [Solveur](operateurs/solveur.md#factorisation-réutilisable-cache-transparent) |

Les matrices d'un même genre assemblées sur un même modèle partagent en plus
leur structure (quels termes existent, et où) : chacune ne garde en propre que
ses valeurs.

## Sauvegarde

Une sauvegarde écrit les objets demandés et tout ce dont ils dépendent ; la
relecture rend le même partage : deux champs posés sur un support restent,
relus, deux champs sur un seul support. Les caches et les compteurs ne sont pas
écrits, ils se recalculent. Le mode d'emploi est au chapitre [Sauvegarde et
relecture](sauvegarde.md), la mécanique [plus bas](#sauvegarde--un-handle-devient-un-numéro).

## Calculs plus gros que la RAM

Sous Linux et macOS, la roue Python peut placer ses grosses allocations dans
des fichiers que le système écrit sur disque quand la mémoire manque : un swap,
sans droit root. On l'active avant de lancer Python par la variable
d'environnement `PYRUCAST_SPILL_DIR`, et `pyrucast.spill_stats()` dit ce qui a
débordé. Voir [Calculs plus gros que la
RAM](operateurs/solveur.md#calculs-plus-gros-que-la-ram).

## Côté Rust

Rust n'a pas de ramasse-miettes : chaque valeur a un propriétaire, et elle est
détruite quand celui-ci disparaît (fin du bloc, ou destruction de la structure
qui la contient). Pour qu'un maillage serve à plusieurs champs, il faut donc un
outil explicite : `Handle<T>`.

### `Handle<T>` : un objet partagé

```rust,ignore
{{#include ../../src/handle.rs:declaration}}
```

Un `Handle` assemble deux briques :

- **`Arc`** tient un compteur de propriétaires. Cloner un `Arc` ne copie pas
  l'objet, cela augmente le compteur ; quand le dernier `Arc` disparaît,
  l'objet est détruit.
- **`RwLock`** (de la bibliothèque `parking_lot`) est un verrou
  lecture/écriture : plusieurs lecteurs à la fois, ou un seul écrivain.

Chaque objet Python enveloppe un `Handle`, ou pour un agrégat la liste des
handles de ses zones : une variable Python est un propriétaire comme un autre.

```rust,ignore
{{#include ../../tests/doc_memoire.rs:partage}}
```

Il en découle :

- **La libération est automatique.** Quand le dernier handle disparaît, la
  fonction de destruction de l'objet (`Drop`) s'exécute, une fois : c'est là
  qu'une zone de maillage rend ses nœuds au `Coords`.
- **Aucune référence n'est périmée.** Tenir un handle garde l'objet en vie :
  `read` et `write` ne peuvent pas échouer.
- **L'identité, c'est l'adresse.** `same_object` dit si deux handles désignent
  le même objet. L'union `|` s'en sert pour ne pas ajouter deux fois une zone,
  un champ pour reconnaître son support.
- **Il n'y a pas de cycle.** Un objet ne désigne que ce sur quoi il est
  construit (un champ son support, une zone son `Coords`), jamais ce qui le
  tient. Le comptage suffit donc à tout libérer.

L'enveloppe porte ses propres méthodes (`read`, `write`, `same_object`) et
n'expose pas celles de l'`Arc` qui permettraient de contourner le verrou.

### Lire et écrire : les guards

```rust,ignore
{{#include ../../tests/doc_memoire.rs:guards}}
```

`read()` et `write()` rendent un **guard** : la preuve qu'on tient le verrou. Il
s'utilise comme une référence vers l'objet et rend le verrou quand il
disparaît, à la fin du bloc ou par `drop` : on ne peut pas oublier de
déverrouiller.

Un guard tient lui-même l'objet en vie : une fonction peut le rendre, une
structure peut le garder. C'est ainsi que `FieldView` garde ouvertes les zones
d'un champ, et que les opérateurs lisent valeurs et connectivités **en place**,
sans les recopier (voir [Zéro-copie](developper/parallelisme.md#zéro-copie)).

### Deux règles de verrouillage

Le verrou couvre **un seul objet** : deux threads qui travaillent sur des
objets différents ne s'attendent jamais, et plusieurs threads lisent le même
maillage pendant un assemblage. Deux règles, que le compilateur ne vérifie pas,
évitent que deux attentes se bloquent l'une l'autre pour toujours :

1. **Ne pas reprendre un objet qu'on tient déjà, même en lecture.** Le verrou
   ne reconnaît pas le thread qui le tient : un second `read()` attend comme
   n'importe qui, et si un écrivain s'est mis en attente entre les deux, le
   second lecteur attend l'écrivain, qui attend le premier lecteur.
2. **Verrouiller une zone avant son `Coords`**, jamais l'inverse.

Python n'est pas concerné : chaque appel prend et rend ses verrous avant de
revenir.

### Afficher un handle

Afficher un handle donne `<SubMesh #7f3a2c>` : le type et la fin de l'adresse
(ses 24 bits de poids faible), le repère que `print` montre en Python. L'affichage ne lit pas
l'objet, il ne bloque donc jamais, même pendant qu'un guard en écriture est
tenu. Le repère distingue les objets vivants au même instant ; une adresse
libérée sert à nouveau, ce n'est donc pas une identité durable.

### Le compteur par nœud

Un nœud n'est pas un objet : c'est un numéro, `NodeId` (4 octets), qui désigne
une ligne des tableaux du `Coords`. Un handle par nœud coûterait un pointeur
compté par occurrence dans les connectivités — deux millions pour 100 000
`HEX20` — et disperserait en mémoire un tableau contigu. Le `Coords` tient donc
un compteur par nœud, un `Vec<u32>` à côté des positions : un `Node` l'augmente
à sa création et le diminue à sa destruction, une zone de maillage fait de même
pour chaque occurrence de ses mailles.

### Les caches : `OnceLock`

Un `OnceLock` est une case qui se remplit une fois. Le premier qui la lit la
calcule, les suivants lisent le résultat ; si deux threads la calculent en même
temps, un seul résultat est gardé. La remplir ne demande qu'un accès en lecture
à l'objet, un cache se construit donc sous un `read()`. La vider demande un
accès en écriture : c'est ce que font les opérations qui modifient l'objet.

| Objet | Membre | Type | Vidé par |
|---|---|---|---|
| `SubMesh` | `poi1_companion` | `OnceLock<Handle<SubMesh>>` | toute modification de la connectivité |
| `SubMesh` | `node_index` | `OnceLock<HashMap<NodeId, usize>>` | toute modification de la connectivité |
| `SubFiniteElementSpace` | `coloring` | `OnceLock<Vec<Vec<usize>>>` | rien : les zones d'un espace sont scellées |
| `Model` | `matrix_patterns` | un `OnceLock<Arc<AssemblyPattern>>` par genre de matrice | `add_sub`, `add_subs` |
| `Matrix` | `factorization` | `Mutex<Option<…>>` | tout changement de la matrice |

La factorisation est dans un `Mutex`, un verrou simple, plutôt que dans un
`OnceLock` : le solveur peut la remplacer. Tous ces membres portent
`#[serde(skip)]`, ce qui les exclut de la sauvegarde.

Ce qui ne change plus se partage sans verrou, par un `Arc` seul : le motif
d'un modèle, et la structure des matrices assemblées sur lui (numérotation,
position des termes). Une matrice assemblée ne possède que ses valeurs.

### Sauvegarde : un handle devient un numéro

Le trait `Portable` (`serde` + `bincode`) fixe les octets d'un objet, les mêmes
sous Linux et Windows (voir
[Sérialisation](conventions.md#sérialisation--un-seul-mécanisme)).

Un handle est une adresse, qui n'a pas de sens dans un autre processus. Ce qui
part sur le disque est donc un numéro **propre au fichier** ; la table qui
traduit l'un en l'autre n'existe que pendant un `save` ou un `load`, et
sérialiser un handle hors de là est une erreur.

Rien ne déclare ce qu'un objet référence : c'est la sérialisation d'un handle
qui le découvre.

```text
déjà vu ?  → écrire son numéro, fini
sinon      → réserver un numéro
             sérialiser l'objet dans son propre enregistrement
                (ce qui découvre, récursivement, ses propres dépendances)
             déposer l'enregistrement
             écrire le numéro
```

Ajouter un membre `Handle` à une structure suffit à le faire sauvegarder,
sans liste à tenir à jour ailleurs. Les enregistrements sortent dans l'ordre
des dépôts, toute dépendance avant ce qui la référence : la relecture est une
simple boucle. Rencontrer un numéro réservé mais pas encore déposé signalerait
un cycle ; l'écriture s'arrête alors sur une erreur qui nomme l'objet.

À la relecture, chaque objet passe par `on_load` avant d'être rendu : c'est là
qu'il reconstruit ce qui n'est pas écrit. Les compteurs des objets se refont
seuls, chaque handle relu en étant un. Ceux des nœuds repartent de zéro : le
`Coords` relu les remet à zéro, puis chaque zone de maillage relue compte ses
occurrences.
