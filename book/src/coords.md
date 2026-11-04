# Coordonnées (Coords)

Un `Coords` est le **magasin des nœuds** : il leur donne une identité et des
coordonnées, en dimension fixée, dans **une ou plusieurs configurations**
(référence, déformée…). C'est le premier objet de pyrucast — maillages, champs
et espaces éléments finis s'y rattachent. On y crée les nœuds, et chacun se
manipule par un [`Node`](node.md), l'objet du chapitre suivant.

```python
{{#include ../../tests/python/test_doc_coords.py:creation}}
```

## Identité et durée de vie des nœuds

Chaque nœud créé reçoit un identifiant (`node.id`) **stable** et **jamais
réutilisé**, valable pour toute la vie du `Coords`. C'est par lui que les
maillages et les champs désignent leurs nœuds.

Un nœud ne se supprime pas : il n'existe aucune méthode pour le retirer, et ses
coordonnées restent lisibles tant que le `Coords` existe. `refcount(id)` dit
combien de références le désignent — chaque `Node` qui le tient, chaque maille
qui le contient — et `acquire(id)` rend un nouveau `Node` pour un identifiant
existant. Ces deux méthodes prennent un **identifiant brut**, pas un `Node` :
passer un `Node` ajouterait la référence même qu'on cherche à compter.

## Repère de révolution

Un `Coords` déclare aussi **comment lire ses coordonnées** : cartésien par
défaut, ou **axisymétrique** — le plan méridien \\( (r, z) \\) d'un solide de
révolution, avec \\( x = r \ge 0 \\) (rayon) et \\( y = z \\) (axe). C'est
l'équivalent du `OPTI MODE AXIS` de Cast3M, et sa place est bien la géométrie :
le repère change la **mesure d'intégration** elle-même,

\\[
d\Omega = 2\pi r \\, |J| \\, d\xi,
\\]

donc rigidité, masse, conductivité, flux réparti, volumes et forces internes
d'un coup — sur le corps comme sur ses bords. Le facteur \\( 2\pi \\) est celui
de l'**anneau complet** : les masses, volumes et résultantes nodales sont ceux de
la pièce de révolution entière.

```python
{{#include ../../tests/python/test_doc_coords.py:axisymetrique}}
```

Un rayon négatif est refusé par `add_node` comme par `set_position`, plutôt que
de ressortir en `|J|` négatif au fond d'une intégrale. Tout espace éléments
finis bâti sur ces `Coords` hérite du repère, si bien qu'un corps et son bord
ne peuvent pas diverger.

Côté **mécanique**, l'axisymétrie ajoute la déformation orthoradiale
\\( \varepsilon_{\theta\theta} = u_r/r \\), qui relève du modèle et non de la
géométrie : voir [Élasticité linéaire](mecanique/elasticite.md#théorie). La
**thermique** n'a rien à changer.

Côté **visualisation**, un tracé axisymétrique montre par défaut la section
méridienne ; l'option `revolve` de `plot` la balaie pour dessiner le corps de
révolution lui-même — voir
[Visualisation](visualization.md#axisymétrie--section-méridienne-ou-corps-de-révolution).

## Plusieurs configurations

Une configuration est un jeu complet de coordonnées pour les mêmes nœuds :
référence, déformée, prédite. `add_config(nom)` clone la configuration
**active** sous un nouveau nom et rend son indice ; `select(i)` change la
configuration active, et `node.position()` lit toujours dans celle-là.

```python
{{#include ../../tests/python/test_doc_coords.py:configurations}}
```

Pour **calculer** dans une autre configuration — une matrice sur la déformée,
une normale sur la paroi telle qu'elle est —, `with coords.in_config(i):`
l'active le temps du bloc, puis rend la configuration d'avant, même si le bloc
lève une exception. C'est la forme à préférer à une paire `select(i)` …
`select(0)`, qui laisse la mauvaise configuration active à la première
exception. Les blocs s'imbriquent. La configuration active appartient au
`Coords`, pas au bloc : un autre fil qui calcule sur le même `Coords` pendant
le bloc la voit aussi.

```python
{{#include ../../tests/python/test_doc_coords.py:in_config}}
```

## Permutation pour le solveur

Une permutation optionnelle fixe l'**ordre des nœuds pour le solveur** sans
toucher à leur identité : `permutation[id]` est la place du nœud `id`, et les
matrices assemblées rangent leurs degrés de liberté dans cet ordre. Elle se
pose à la main par `set_permutation`, avec une entrée par nœud créé
(`capacity`) et aucune valeur en double ; `clear_permutation` revient à
l'ordre des identifiants.

```python
{{#include ../../tests/python/test_doc_coords.py:permutation}}
```

## Inspecter un `Coords`

`dim`, `is_axisymmetric`, `names()` et `active` décrivent le `Coords` ;
`print(c)` le résume en une ligne, `repr(c)` en donne la structure et
`c.dump()` la table complète des nœuds, coordonnées et références — les
[trois niveaux de listing](aspect-informatique.md#trois-niveaux-de-listing).

## Pourquoi plusieurs configurations dans un seul `Coords`

L'autre conception possible serait un `Coords` par configuration. pyrucast
garde **un seul jeu d'identités par domaine**, et plusieurs configurations
dedans :

| | Plusieurs configurations dans un `Coords` (pyrucast) | Un `Coords` par configuration |
|---|---|---|
| identité | l'id 42 désigne le **même nœud** dans toutes les configurations ; maillages et champs restent valides quelle que soit la configuration active | l'id 42 de deux objets désigne deux nœuds différents : tout maillage ou champ à cheval doit porter une table de correspondance |
| changer de configuration | un changement d'indice, rien à renuméroter | un autre objet, et ce qui le référence à reconstruire |
| limite | toutes les configurations ont les **mêmes nœuds** | des jeux de tailles différentes sont possibles |

Des `Coords` distincts servent aux domaines vraiment indépendants.

## Côté Rust

En Rust, un `Coords` se partage par un `Handle<Coords>`, une référence comptée
munie de son verrou, et chaque nœud porte en plus son propre compteur dans le
`Coords` : c'est lui que `refcount` lit, et que `Node`, les maillages et les
champs incrémentent. Le détail est dans [Modèle mémoire](memory-model.md).
Les mêmes opérations s'écrivent ainsi :

```rust,ignore
{{#include ../../tests/doc_coords.rs:repere}}
```

```rust,ignore
{{#include ../../tests/doc_coords.rs:configurations}}
```

`in_config` y reçoit le bloc en fermeture, l'idiome Rust du `with` Python :

```rust,ignore
{{#include ../../tests/doc_coords.rs:in_config}}
```

```rust,ignore
{{#include ../../tests/doc_coords.rs:permutation}}
```
