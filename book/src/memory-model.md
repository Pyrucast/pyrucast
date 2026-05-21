# Modèle mémoire

Ce chapitre décrit comment pyrucast gère la mémoire : principes directeurs, implémentation actuelle, et évolutions prévues quand le besoin se mesurera.

## Principes directeurs

1. **Une seule « pile » d'objets, à la cast3m.** Tous les objets vivent dans un store global au processus, indexé en interne par `TypeId` (un slab par type). L'accès passe par des fonctions de module — pas de Session à passer — mais avec la sûreté de typage de Rust. La complexité de la gestion mémoire est confinée à `src/store.rs` ; le reste du code reste simple.

2. **Libération automatique, sans `remove` explicite.** Toute référence à un objet passe par un `Handle<T>` qui est **compté** (`Clone` incrémente, `Drop` décrémente) et **générationnel** (un slot recyclé invalide les anciens handles). Quand le dernier handle disparaît, le slot est libéré.

3. **Refcount à deux niveaux.** Le store gère la durée de vie des *slots* — un objet entier est-il logiquement vivant ? À l'intérieur d'objets composites comme `Configuration`, un second niveau de refcount gère la durée de vie de *composants* internes (les nœuds). Voir le chapitre [Configuration](configuration.md). Le ramasse-miettes manuel (`gc()`) opère sur ce second niveau.

4. **Swap transparent vis-à-vis de `Drop`.** Un slot peut être évincé sur disque sans que `Drop` ne s'exécute (l'objet reste *logiquement* vivant, juste relocalisé). Au décrément final, le slot est rechargé depuis le disque avant `Drop` si nécessaire, de sorte que les effets de bord s'exécutent **exactement une fois** sur la durée de vie de l'objet. Cela rend safe la combinaison « objets à `Drop` avec effets de bord » + « swap ».

5. **Sérialisation portable comme socle unique.** Le trait `Persist` (`serde` + `bincode`) est l'unique mécanisme de sérialisation. Il sert au swap **et** à la sauvegarde utilisateur. Le format binaire est identique Linux ↔ Windows (cf. [Conventions](conventions.md)).

## Implémentation actuelle

### API par fonctions de module

```rust,ignore
use pyrucast::store::{insert, with, with_mut, swap_out, compact};

let h = insert(mon_objet);              // dépose dans le store, renvoie un handle
with(&h, |o| { /* lecture */ }).unwrap();
with_mut(&h, |o| { /* écriture */ }).unwrap();
swap_out(&h).unwrap();                  // évince sur disque, libère la RAM
compact::<MonObjet>();                  // rétrécit la mémoire en queue de Vec
```

Côté Python, les bindings sont ajoutés objet par objet (Phase 2) ; le module expose en plus `pyrucast.set_swap_dir(path)` et `pyrucast.swap_dir()` pour configurer le répertoire de swap.

### `Handle<T>` générationnel et compté

`Handle<T>` est une struct comportant :

- un **index** de slot dans le store du type `T` ;
- une **génération** : un slot recyclé incrémente sa génération, rendant tout ancien handle automatiquement obsolète (l'accès renvoie `PyrucastError::StaleHandle`) ;
- un compteur de références implicite : `Clone` incrémente, `Drop` décrémente. Quand le refcount atteint 0, le slot est marqué libre, et la valeur résidente est droppée **hors du verrou interne** pour éviter toute rentrance.

### États du slot

```text
        ┌─────────────────────┐
        │      Resident       │  ◀──┐
        │   (valeur en RAM)   │     │  rechargement automatique
        └──────┬──────────────┘     │  au prochain with / with_mut
   swap_out()  │                    │
               ▼                    │
        ┌─────────────────────┐     │
        │       OnDisk        │  ───┘
        │ (fichier .bin sur   │
        │     disque)         │
        └──────┬──────────────┘
   refcount 0  │
               ▼
        ┌─────────────────────┐
        │        Free         │
        │ (slot recyclable    │
        │  via la free-list)  │
        └─────────────────────┘
```

Le format binaire posé sur disque est celui du trait `Persist`, partagé avec la sauvegarde/relecture utilisateur.

### Sûreté du swap vis-à-vis de `Drop`

Beaucoup d'objets pyrucast portent des effets de bord dans leur `Drop` (ex. `SubMesh` décrémente le refcount des nœuds de la `Configuration`). Le store assure :

- `swap_out` n'exécute **pas** le `Drop` de la valeur évincée (`std::mem::forget` interne) — l'objet est logiquement vivant, juste relocalisé.
- Au décrément final du refcount, si le slot est `OnDisk`, on recharge depuis le disque avant de dropper. `Drop` s'exécute donc une et une seule fois sur la durée de vie de l'objet, quel que soit le parcours swap.

### Fragmentation et compactage

- Les slots libres sont enchaînés dans une **free-list** et réutilisés en priorité par `insert` : les indices restent stables dans le temps.
- `compact::<T>()` retire les slots libres en **queue** de `Vec` et rétrécit la mémoire allouée. Le compactage ne déplace pas les slots vivants : les handles existants restent valides.

### Concurrence

Le store interne de chaque type `T` est protégé par un `Mutex`. Les opérations sur des **types différents** sont indépendantes. Une seule contrainte d'usage : **ne pas appeler `insert`/`with`/`with_mut`/`swap_out`/`compact` sur le même type `T` à l'intérieur d'un closure passé à `with` ou `with_mut`** — la rentrance sur le même mutex provoquerait un interblocage.

### Pourquoi pas d'ownership Rust direct ?

Les exigences combinées — auto-libération, gestion de fragmentation, swap sur disque, renumérotation des nœuds indépendante de leur identité — ne s'expriment pas naturellement avec `Arc`/`Rc`/`Weak`. Un store dédié permet de découpler **identité de l'objet** et **placement physique**, et de partager un seul mécanisme entre swap mémoire et sauvegarde fichier.

### Pourquoi un refcount plutôt qu'un mark-and-sweep ?

Une alternative tentante au refcount par nœud serait un GC **mark-and-sweep** : pas de compteur, mais à chaque appel à `gc()`, parcourir tous les `Node`/`SubMesh`/`NodeField` vivants pour marquer les `NodeId` atteignables. La simplification est apparente ; deux obstacles s'y opposent dans notre architecture.

**Obstacle 1 — où sont les racines ?**

Les `SubMesh`, `Mesh`, `NodeField` (à venir) vivent dans le store : énumérables. Mais un `Node` utilisateur vit **sur la pile Rust** (ou dans le heap Python via PyO3), pas dans le store. Pour qu'un `Node` se signale à la `Configuration` comme racine vivante, il n'y a que deux options :

1. Maintenir une **liste séparée** des `Node` vivants dans la `Configuration`, mise à jour à `Clone`/`Drop`. C'est un refcount déguisé en `HashSet<NodeId>` — sans gain.
2. **Changer le contrat** : `Node` devient une vue non-protectrice, et seul un objet du store (typiquement un `SubMesh` POI1) peut maintenir un nœud vivant. C'est exactement le modèle cast3m (« un point isolé n'existe pas en dehors d'un MAILLAGE »). Légitime, mais c'est une rupture d'API plus large que la simplification cherchée.

**Obstacle 2 — le swap se paie cher**

Marquer les nœuds atteignables impose de parcourir **tous les `SubMesh` du store**, y compris ceux qui sont `OnDisk`. Il faudrait donc les recharger en mémoire à chaque `gc()` — ce qui défait l'intérêt du swap. Le refcount actuel, lui, est **sérialisé avec le `SubMesh`** et reste cohérent même quand l'objet est sur disque ; aucun rechargement n'est nécessaire à la collecte.

**Où est vraiment le coût du refcount ?**

- `SubMesh::add_cell` : `N × incref` sous **un seul** verrou — négligeable comparé à la création de la cellule.
- `Node::clone`/`drop` : un verrou + une indexation `Vec<u32>`. Sensible *seulement* si on clone/drop beaucoup de `Node` en boucle serrée (rare en FE — on construit le maillage, puis on n'y touche plus).
- **Complexité conceptuelle** : la logique de rollback dans `add_cell` et la doctrine « deux niveaux de refcount » pèsent plus sur la lisibilité que sur le CPU.

**Choix pyrucast** : refcount pour l'instant. Si la complexité devient gênante, le levier le plus rentable n'est **pas** de remplacer le mécanisme mais de **simplifier le contrat de `Node`** (option 2 ci-dessus, mode cast3m-pur) — cela supprime à la fois le refcount *et* la logique de rollback, sans nécessiter de recharger les objets swappés. À garder en tête pour une éventuelle Phase 7 si on observe que les `Node` « loose » sont peu utilisés en pratique.

## Limites connues et évolution prévue

Le store actuel est une fondation correcte mais il a trois angles morts, déjà connus de l'histoire de cast3m.

### Les trois angles morts

1. **Fragmentation interne du `Vec`.** La free-list réutilise les trous, mais les slots libérés *au milieu* d'un `Vec` ne rendent pas la mémoire à l'OS. `compact()` ne rogne que la queue. Sur des cycles intensifs de création/destruction, la mémoire ne redescend jamais sous le high-water mark.

2. **Pas de politique d'éviction.** `swap_out` est manuel. Aucune notion intégrée de « ces objets sont en cours d'utilisation, ces autres sont transitoires » : c'est à l'appelant de décider quoi évincer et quand.

3. **Fragmentation inter-types.** Chaque type a son propre `Vec` ; aucune mutualisation des allocations, aucun renvoi automatique au système.

### Trois pistes pour aller plus loin

| Approche | Idée | Coût | Bénéfice |
|---|---|---|---|
| **A. Indirection + compactage déplaçant** | Table `id → slot_idx`. On déplace les slots vivants pour combler les trous, la table est mise à jour. Les `Handle` restent valides car ils référencent l'`id` logique, pas le slot physique. | 1 indirection supplémentaire par accès ; ~100 lignes. | Fragmentation interne **résolue**. C'est l'approche historique de cast3m. |
| **B. Swap annoté + éviction LRU** | Chaque slot porte une priorité (`Pinned` / `Working` / `Scratch`) et un `last_used`. Au-dessus d'un budget RAM, on évince automatiquement le bas-priorité + le plus ancien. | Heuristiques, instrumentation, paramètres à régler. | Swap « intelligent » façon cast3m moderne : plus besoin d'appels manuels à `swap_out`. |
| **C. Arènes par génération** | Jeune génération (objets transitoires) collectée souvent ; vieille génération (Configuration, Mesh ancrés) collectée rarement. Inspiré des GC générationnels. | Lourd à implémenter (semi-spaces, indirection, write-barrier). | Très bien adapté au profil FE — beaucoup de scratch + peu d'objets stables. Le plus ambitieux. |

### Stratégie d'évolution

Le store actuel reste la **bonne fondation** — on ne le remplacera pas. Les évolutions ci-dessus seront ajoutées **quand le besoin se mesure**, pas par anticipation. La séquence prévue, à confirmer par les premières mesures de Phase 6 (durcissement) :

1. **D'abord (A)** : indirection + compactage déplaçant, quand des cycles intensifs montreront concrètement des hauts plateaux mémoire. ~100 lignes, ne change pas l'API publique.
2. **Ensuite (B)** : priorité + éviction automatique, le jour où le swap manuel deviendra insuffisant (typiquement sur les premiers solveurs sur gros maillage).
3. **(C) à arbitrer** : on n'attaquera la généralisation arènes/générations qu'une fois que A et B auront révélé leurs propres limites. L'expérience cast3m suggère que les bons paramètres se trouvent par mesure, pas par design *a priori*.

L'information **structurelle** (qui référence quoi) est déjà encodée via la composition des `Handle` ; le swap intelligent (B) pourra s'en servir sans bookkeeping additionnel.
