# Champ (`Field` / `SubField`)

Les deux familles de champs de pyrucast — le [champ aux nœuds](node-field.md)
et le [champ aux points de Gauss](element-field.md) — partagent un **contrat
commun**, capturé par deux traits Rust (`src/containers/field.rs`) :

- **`SubField`** — une **zone** : un bloc homogène de valeurs, des composantes
  nommées + un buffer plat ;
- **`Field`** — le niveau **agrégat**, *blanket-implémenté* pour tout
  [`Aggregate`](aggregate.md) dont la zone est un `SubField`.

Ce chapitre décrit ce contrat (composantes, statistiques, arithmétique) ; les
deux chapitres suivants donnent les spécificités de chaque famille.

## Principe du trait

### `SubField` : un bloc homogène

Une zone porte :

- une liste ordonnée de **composantes nommées** (`"UX"`, `"UY"`, `"T"`, `"k"`,
  `"sigma_xx"`, …) — au moins une, noms uniques ;
- un **buffer plat** de `f64` dans lequel l'**indice de composante varie le
  plus vite** (stride = nombre de composantes).

Le contrat est purement **structurel** : peu importe que les « lignes » du
buffer soient des nœuds (`SubNodeField`) ou des couples `(cellule, point de
Gauss)` (`SubElementField`) — dès qu'une zone fournit ses composantes et son
buffer, le trait en dérive le reste : recherche d'une composante par nom,
`min`/`max` par composante, opérations scalaires.

```text
   SubField (composantes [c0, c1], stride = 2)
   values = [ ligne0.c0, ligne0.c1, ligne1.c0, ligne1.c1, … ]
                                     └─ composante varie le plus vite ─┘
```

Chaque zone connaît aussi son **support** (`support()` : un `Handle<SubMesh>`
pour un champ aux nœuds, un `Handle<SubFiniteElementSpace>` pour un champ aux
Gauss). Deux zones sont « sur le même support » (`same_support`) si leurs
handles désignent le **même slot** — c'est la précondition pour les combiner.

### `Field` : le repli sur les zones

Au niveau agrégat, `Field` **replie** les opérations sur les zones :

- `components()` — l'**union** des composantes des zones (ordre de première
  apparition) ; une composante peut n'exister que sur certaines zones ;
- `min(c)` / `max(c)` — repliés sur les zones qui définissent `c` (erreur si
  aucune) ;
- `view()` — une vue **zéro-copie** (un guard de lecture par zone), utilisée
  par les opérateurs qui font beaucoup de lectures (gradient, solveur, viz).

## Opération arithmétique

L'arithmétique des champs se décline en trois familles, du plus simple au plus
contraint.

### 1. Scalaire (broadcast)

`field + s`, `field - s`, `field * s`, `field / s`, `field ** s` renvoient un
**nouveau** champ où l'opération est appliquée à **toutes** les valeurs de
toutes les composantes de toutes les zones. Disponible au niveau **zone**
(`SubField`) **et** au niveau **agrégat** (`NodeField` / `ElementField`,
dunders Python `__add__`, …, `__pow__`).

```python
scaled = mat * 1.1  # nouveau champ, toutes composantes × 1.1
shifted = u - 5.0  # nouveau champ
energy = u**2.0  # puissance élément par élément (exposant fractionnaire OK)
```

> `+=` n'est **pas** surchargé : `f + s` ne mute pas `f`. (Côté Rust, la
> version consommante est zéro-copie, la version par référence clone d'abord.)

> **Puissance — Python seulement.** Rust n'a pas d'opérateur de puissance ;
> `+`/`-`/`*`/`/` passent par `Add/Sub/Mul/Div`, mais `**` est exposé par le seul
> dunder `__pow__`. Côté Rust, faire une puissance via les primitives génériques :
> `combine_scalar(|a, b| a.powf(b), s)` (scalaire) ou
> `merge_components(other, |a, b| a.powf(b))` (binaire). La forme ternaire `pow(x, y, z)`
> (modulo) est refusée — elle n'a pas de sens sur des flottants.

### 2. Par composante (en place)

Pour ne toucher **qu'une** composante, sur toutes les zones qui la portent :
`add_to_component(c, s)`, `sub_to_component`, `mul_to_component`,
`div_to_component` — **en place**, erreur seulement si **aucune** zone ne
définit `c` (la division par zéro est refusée). Au niveau zone, `set_uniform(c,
v)` force une composante à une valeur constante.

```python
mat.mul_to_component("E", 0.95)  # ne met à l'échelle que "E"
```

### 3. Binaire entre champs

Combiner **deux champs** valeur à valeur se décline selon le niveau.

Les **opérateurs** `+ - * /` (et `**`), aussi bien de **zone à zone**
(`merge_components`) que d'**agrégat à agrégat** (`merge_field`), sont **par (support,
composante)** en **union avec passthrough** — les deux opérandes n'ont **pas**
besoin du même jeu de composantes ni de la même décomposition :

- la sortie couvre l'**union** des supports des deux opérandes ;
- sur un support porté des deux côtés, les zones sont fusionnées composante par
  composante (`merge_components`) : une composante définie **des deux côtés**
  devient `op(a, b)` ; une composante d'un **seul** côté **passe telle quelle**
  (passthrough **brut**, pour tous les opérateurs — donc `a - b` sur une
  composante propre à `b` vaut `b`, pas `-b`) ;
- un support porté d'un **seul** côté voit sa (ses) zone(s) passer **inchangées**.

Cela suppose l'invariant de champ (au plus une zone par `(support, composante)`,
garanti par l'union `|`) ; la sortie l'hérite par construction.

> **« Même support » = le même objet, pas la même géométrie.** L'appariement se
> fait par **identité d'objet** (`Handle::same_object`, via
> `SubField::same_support`), jamais en comparant les nœuds. Deux supports
> distincts qui portent les mêmes nœuds — ou qui n'en partagent que quelques-uns,
> comme deux régions de bord adjacentes assemblées chacune de son côté — comptent
> donc chacun comme « porté d'un seul côté » : les deux zones passent
> **inchangées** et se retrouvent côte à côte dans la sortie. Rien n'est sommé au
> nœud commun, et à la lecture agrégat (`gather`, `value`) c'est la **première**
> zone définissant `(nœud, composante)` qui l'emporte — l'autre contribution est
> perdue sans erreur. Pour additionner deux régions qui se touchent, il faut donc
> les ramener sur un support **partagé** au préalable :
> `restrict(a, m) + restrict(b, m)` (même maillage `m` ⇒ même support `POI1`
> canonique, mis en cache), ou `restrict_like(b, a)` pour retomber sur celui de
> `a` (cf. [Opérateurs sur les champs](operateurs/champs.md)).

Tous les opérateurs passent par `merge_components` (zone) / `merge_field`
(agrégat) / `merge_subfield` (maj ciblée d'une zone). Là où un jeu de composantes
divergent est une **erreur** plutôt qu'un passthrough (p. ex. l'interpolation
`Evolution` entre deux valeurs tabulées du *même* champ), on garde `merge_components`
mais précédé du garde-fou `SubField::check_same_components` (même support **et**
même jeu de composantes, sinon erreur).

La même mécanique vaut pour `field ** field` (puissance élément par élément,
exposant pris dans le second champ). La division — et la puissance à exposant
fractionnaire sur base négative — ne se protègent **pas** des cas limites à ce
niveau (sémantique numpy : `inf` / `nan`).

### 4. Fonctions unaires (`cos`, `exp`, …)

Des fonctions mathématiques de base s'appliquent **élément par élément**,
renvoyant un **nouveau** champ de même type (zone ou agrégat). Style numpy,
exposées **au top-level** côté Python :

```python
import pyrucast as pc

champ2 = pc.field.cos(champ1)  # cosinus de chaque valeur
e = pc.field.exp(-pc.field.abs(u))  # elles se composent librement
norme = pc.field.sqrt(sx**2.0 + sy**2.0)
```

Jeu disponible : `abs`, `sqrt`, `exp`, `log` (népérien), `log10`, `cos`,
`sin`, `tan`, `sinh`, `cosh`, `tanh`. Sémantique **non gardée** comme le reste
(`log` d'un négatif → `nan`). Côté Rust ce sont des fonctions nommées
(`ops::field::cos(&f)`, …, génériques via le trait `MapValues`) ; il n'y a pas
de syntaxe `cos(x)` pour un opérateur en Rust, donc seul Python en profite à
l'écriture. Tout repose sur la primitive `map_all` (cf. plus haut), sans
logique nouvelle.

## Interface (résumé)

| Niveau | Méthode | Effet |
|---|---|---|
| zone & agrégat | `components()` | composantes (union au niveau agrégat) |
| zone & agrégat | `min(c)` / `max(c)` | extrema d'une composante |
| zone | `set_uniform(c, v)` | force `c` à `v` |
| zone & agrégat | `f + s`, `f - s`, `f * s`, `f / s` | scalaire, nouveau champ |
| zone & agrégat | `f ** s`, `f ** g` | puissance élément par élément (Python `**` ; Rust : `merge_components`) |
| zone & agrégat | `add_to_component(c, s)` … | scalaire sur une composante, en place |
| zone & agrégat | `f + g`, `f - g`, `f * g`, `f / g` | opérateurs binaires : union par `(support, composante)`, passthrough brut |
| zone | `merge_components(other, op)` | primitive des opérateurs de zone : union par composante, passthrough brut |
| zone | `check_same_components(other)` | garde-fou : erreur si support/composantes divergent (à appeler avant `merge_components` quand un écart est un bug) |
| agrégat | `merge_field(other, op)` | primitive des opérateurs d'agrégat : union par `(support, composante)` |
| agrégat | `merge_subfield(sub, op)` | binaire ciblé sur une zone (union/passthrough) |

Les deux familles concrètes ajoutent leurs accès indexés et leurs
constructeurs propres :

- [`NodeField` / `SubNodeField`](node-field.md) — valeurs par nœud, lecture
  agrégat `field.value(node, "c")`, écriture par zone ;
- [`ElementField` / `SubElementField`](element-field.md) — valeurs par
  `(cellule, point de Gauss)`.

> **Et l'union `|` ?** L'union compose des **zones** (structure), l'arithmétique
> combine des **valeurs**. Pour un `ElementField`, l'union ne **fusionne plus**
> les zones : elle **valide** simplement qu'aucune composante n'est portée deux
> fois sur le même support (deux zones de même support à composantes disjointes
> restent côte à côte). Pour un `NodeField`, l'union finalise encore en fusionnant
> les zones de même support (et lève si elles divergent sur une valeur partagée).
> La fusion explicite reste offerte par `node_field.consolidate` / `element_field.consolidate`
> (cf. [Opérateurs sur les champs](operateurs/champs.md)).
