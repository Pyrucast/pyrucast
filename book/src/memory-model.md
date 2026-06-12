# Modèle mémoire

Ce chapitre décrit comment pyrucast gère la mémoire : principes directeurs, implémentation actuelle, et évolutions prévues quand le besoin se mesurera.

## Principes directeurs

1. **Une seule « pile » d'objets, à la cast3m.** Tous les objets vivent dans un store global au processus, indexé en interne par `TypeId` (un slab par type). L'accès passe par des fonctions de module — pas de Session à passer — mais avec la sûreté de typage de Rust. La complexité de la gestion mémoire est confinée à `src/store.rs` ; le reste du code reste simple.

2. **Libération automatique, sans `remove` explicite.** Toute référence à un objet passe par un `Handle<T>` qui est **compté** (`Clone` incrémente, `Drop` décrémente) et **générationnel** (un slot recyclé invalide les anciens handles). Quand le dernier handle disparaît, le slot est libéré.

3. **Refcount à deux niveaux.** Le store gère la durée de vie des *slots* — un objet entier est-il logiquement vivant ? À l'intérieur d'objets composites comme `Configuration`, un second niveau de refcount gère la durée de vie de *composants* internes (les nœuds). Voir le chapitre [Configuration](configuration.md). Le ramasse-miettes manuel (`gc()`) opère sur ce second niveau.

4. **Swap transparent vis-à-vis de `Drop`.** Un slot peut être évincé sur disque sans que `Drop` ne s'exécute (l'objet reste *logiquement* vivant, juste relocalisé). Au décrément final, le slot est rechargé depuis le disque avant `Drop` si nécessaire, de sorte que les effets de bord s'exécutent **exactement une fois** sur la durée de vie de l'objet. Cela rend safe la combinaison « objets à `Drop` avec effets de bord » + « swap ».

5. **Sérialisation portable comme socle unique.** Le trait `Persist` (`serde` + `bincode`) est l'unique mécanisme de sérialisation. Il sert au swap **et** à la sauvegarde utilisateur. Le format binaire est identique Linux ↔ Windows (cf. [Conventions](conventions.md)).

## Implémentation actuelle

### Vue d'ensemble : une analogie

Imaginez une **bibliothèque municipale** où chaque rayon est dédié à un type d'ouvrage : un rayon pour les `Configuration`, un autre pour les `SubMesh`, un autre pour les `NodeField`, etc. Chaque rayon est une **étagère numérotée** : la case 0, la case 1, la case 2…

- Quand vous déposez un livre, la bibliothécaire vous remet un **ticket** : « rayon Configuration, case n°7, *édition 3* ». Ce ticket, c'est un `Handle`.
- Pour relire votre livre, vous présentez le ticket à la bibliothécaire : elle vérifie la case et l'édition (anti-falsification) — une formalité d'un instant — puis vous installe à un **pupitre de lecture** attaché à cette case : vous consultez le livre **sur place**, sans l'emporter. Ce pupitre, c'est un *guard*. Plusieurs lecteurs peuvent partager le même livre en même temps ; pour **annoter** le livre en revanche, il faut le pupitre exclusif — on attend que les lecteurs aient fini.
- Pendant que vous lisez la case n°7, le reste du rayon est entièrement libre : chaque case a son propre pupitre, la bibliothécaire ne bloque jamais le rayon entier.
- Quand votre ticket disparaît (et tous ses duplicatas), le livre est jeté et la case redevient libre. Mais la prochaine personne qui dépose un livre dans cette case repartira avec un ticket portant l'**édition 4** — votre vieux ticket édition 3 ne marchera plus, même si la case est la même.

Tout pyrucast tient dans cette image. Les sections suivantes en déroulent les morceaux.

### Le store : un grand tableau par type

Chaque type d'objet (`Configuration`, `SubMesh`, …) possède **son propre store** : un `Vec<Slot<T>>` global au processus, créé à la demande lors du premier `insert::<T>(...)`. Tous les stores sont enregistrés dans une **table globale** indexée par `TypeId`, ce qui permet à `insert::<T>` de retrouver le bon `Vec` à l'exécution. Le code utilisateur n'a pas besoin de connaître cette table : il ne voit que les fonctions `insert`, `read`, `write`, `swap_out`, `compact`.

Un `Slot<T>` se compose d'une **cellule partagée** (le casier proprement dit) et d'un compteur d'édition :

```rust,ignore
struct SlotCell<T> {
    lock: Arc<RwLock<SlotState<T>>>,  // le contenu, derrière SON verrou
    refcount: AtomicU32,              // nombre de Handle vivants
}
struct Slot<T> {
    cell: Arc<SlotCell<T>>,           // boîte partagée (voir « Arc » plus bas)
    gen: u32,                         // génération courante du slot
}
```

`SlotState<T>` vaut `Resident(value)` (l'objet est en RAM), `OnDisk(path)` (évincé sur disque) ou `Free` (case vide).

Visuellement, le store ressemble à ceci :

```text
   store<Configuration> :
   ┌─────┬─────┬─────┬─────┬─────┐
   │ #0  │ #1  │ #2  │ #3  │ #4  │ ← indices de slot
   │Res  │Free │Res  │OnDsk│Res  │ ← état (dans le RwLock de chaque cellule)
   │gen=1│gen=2│gen=1│gen=1│gen=3│ ← génération
   │rc=2 │rc=0 │rc=1 │rc=1 │rc=4 │ ← refcount (atomique, dans la cellule)
   └─────┴─────┴─────┴─────┴─────┘
                  ↑
            free-list : [1]   ← cases libres prêtes à être recyclées
```

Le mutex du store ne protège plus que ce `Vec` lui-même (résolution d'un ticket en cellule, insertion, recyclage) : il n'est tenu que quelques instants, jamais pendant qu'on lit ou modifie un objet. **Le verrouillage des données est par objet**, via le `RwLock` de chaque cellule.

### Le `Handle` : votre ticket d'accès

`insert(value)` ajoute la valeur dans le store et vous rend un `Handle<T>` :

```rust,ignore
pub struct Handle<T> {
    idx: u32,   // numéro de case
    gen: u32,   // édition au moment de la remise du ticket
    cell: OnceLock<Arc<SlotCell<T>>>,  // cache vers la cellule (ignoré par serde)
}
```

Trois choses à retenir sur le handle :

1. **L'identité, c'est `(idx, gen)`** — deux `u32`. C'est tout ce qui est sérialisé ; le champ `cell` n'est qu'un raccourci résolu paresseusement (un handle fraîchement désérialisé démarre avec un cache vide, rempli au premier accès).
2. **`Clone` est un incrément atomique** : `refcount += 1` sur la cellule, sans prendre aucun verrou. Tous les clones pointent **sur la même case** du store. On peut donc cloner un handle même pendant qu'un guard est ouvert sur l'objet.
3. **`Drop` décrémente le `refcount`**. Quand le compteur tombe à 0, la case est rendue à la free-list et la valeur est détruite (hors de tout verrou — ses effets de bord `Drop` touchent d'autres stores). Tout cela est automatique : aucune fonction `remove()` à appeler.

```rust,ignore
let h1 = insert(MaStruct(42));   // case n°7, refcount = 1
let h2 = h1.clone();             // refcount = 2 (même case)
let h3 = h1.clone();             // refcount = 3
drop(h1); drop(h2);              // refcount = 1
drop(h3);                        // refcount = 0 → case libérée, valeur droppée
```

### Recyclage et générations : la sécurité du ticket

Quand un slot atteint `refcount = 0`, son index est poussé dans la **free-list** du store. Au prochain `insert`, plutôt que d'agrandir le `Vec`, le store dépile cet index et y range la nouvelle valeur. Bénéfice : **les indices restent bornés**, la mémoire ne croît pas à chaque cycle création/destruction.

Mais que se passe-t-il si un ancien handle (un ticket périmé) traîne quelque part ? Sans précaution, il pointerait sur une case maintenant occupée par un **autre** objet — un classique bug *use-after-free*. La parade : la **génération**.

Chaque slot porte un compteur `gen`. À chaque recyclage, `gen` est incrémenté. Le handle stocke la génération au moment de sa création :

```text
   Avant :  slot #7  gen=3  →  Handle{idx: 7, gen: 3} ✓ valide
   Drop tous les handles → slot #7 libéré.
   Nouvel insert → slot #7 réutilisé, gen passe à 4.
   Après :  slot #7  gen=4  →  ancien Handle{idx: 7, gen: 3} ✗ invalide
                            →  nouveau Handle{idx: 7, gen: 4} ✓ valide
```

Toute lecture/écriture commence par `validate(idx, gen)` : si la génération du slot ne correspond pas à celle du handle, l'accès renvoie `PyrucastError::StaleHandle`. Le bug *use-after-free* devient une **erreur récupérable**, jamais une corruption silencieuse.

### Pourquoi des `Handle`, et pas des `&Configuration` directs ?

Une question naturelle : pourquoi ne pas simplement passer une référence Rust `&Configuration` ou un `Arc<Configuration>` ? Trois raisons cumulées :

1. **Indépendance identité / placement.** Le store doit pouvoir **déplacer un objet** (vers le disque par swap, plus tard vers une autre case par compactage déplaçant). Une `&Configuration` interdit ce déplacement (le borrow-checker fige l'adresse). Un `Handle` désigne l'identité ; le store gère l'emplacement physique.
2. **Sérialisation.** Un `&Configuration` n'est pas sérialisable. Un `Handle` est juste `(u32, u32)` : on peut le stocker dans un autre objet, le sauvegarder sur disque, et le relire — le pointage logique survit au round-trip. Combiné au swap Drop-safe, un `SubMesh` qui contient un `Handle<Configuration>` traverse le disque sans casser le graphe d'objets.
3. **API uniforme côté Python.** PyO3 a besoin d'objets `Clone + Send + Sync + 'static` côté Rust pour les exposer en classes Python. `Handle<T>` coche ces cases ; `&Configuration` non.

### Accès via `read` / `write` : les guards

Avec un `Handle`, le code utilisateur ne peut pas écrire `handle.dim` directement. Il passe par un **guard** :

```rust,ignore
let cfg = read(&handle)?;            // verrou lecture sur CET objet seul
println!("dim = {}", cfg.dim());     // cfg se comporte comme un &Configuration
drop(cfg);                           // (ou fin de scope) → verrou relâché

write(&handle)?.add_node(&[0.0, 0.0])?;   // verrou écriture, le temps de l'appel
```

Un guard est un objet temporaire qui prouve que vous détenez le verrou. Il se comporte comme une référence vers la donnée (via `Deref`), et **le verrou est relâché automatiquement à sa destruction** (fin de scope) — c'est du RAII, impossible d'oublier de déverrouiller. Le borrow-checker s'applique normalement à travers le guard : pas de mutation pendant une lecture, exclusion `read`/`write` à l'exécution via le `RwLock`.

Que se passe-t-il exactement sur un `read(&h)` ?

```text
Handle { idx: 7, gen: 3 }                     ← votre ticket
        │
        ▼  (1) mutex du store, un instant :
store<T> :  [#0] [#1] ... [#7 {gen: 3 ✓, Arc ──┐}]
        │   (2) clone de l'Arc, mutex relâché  │
        ▼                                      ▼
            Cellule partagée : SlotCell { lock: RwLock(Resident(données)),
        (3) verrou lecture ─────────┘           refcount: 2 }
        ▼
   ReadGuard ── se comporte comme l'objet, lit les données EN PLACE
```

1. Sous le mutex du store, on vérifie que la case 7 est bien en génération 3 et on clone l'`Arc` de la cellule ; le mutex est relâché aussitôt. **`Arc`** (*Atomically Reference Counted*) est une boîte partagée à tickets de propriété : la cloner ne copie pas le contenu, ça crée un second ticket sur la même boîte, qui ne sera libérée qu'au dernier ticket rendu. Grâce à ce ticket, la cellule ne peut pas être désallouée sous vos pieds, quoi qu'il arrive au store pendant votre lecture.
2. On prend le verrou **de cette cellule seulement** — `RwLock` : N lecteurs simultanés, OU 1 écrivain exclusif (la règle du borrow-checker, vérifiée à l'exécution entre threads). Si l'objet était `OnDisk`, il est rechargé au passage.
3. Le `ReadGuard` rendu est **possédé** (`'static`) : il détient son propre Arc et un clone du `Handle`, donc il peut être retourné par une fonction, rangé dans une struct (c'est le mécanisme de `FieldView`, la vue zéro-copie des champs), et il maintient le slot en vie même si vous droppez votre handle entre-temps.

C'est ce caractère « possédé » qui permet aux opérateurs (gradient, solveur, viz) de lire les données **en place** dans le store pendant toute leur boucle, au lieu d'en faire des copies.

### Concurrence : un verrou par objet

Le mutex du store n'arbitre que l'annuaire (`idx/gen` → cellule, insertion, recyclage) — tenu quelques nanosecondes. Toute la concurrence sur les **données** passe par le `RwLock` de chaque cellule :

- deux threads qui manipulent **des objets différents** — même du même type — ne se gênent jamais ;
- plusieurs threads peuvent **lire le même objet** simultanément ;
- un écrivain est exclusif sur **son** objet, et seulement le sien.

C'est la granularité qui prépare le parallélisme interne aux opérateurs (plusieurs threads lisant le même maillage pendant l'assemblage).

**Une seule règle à respecter** : **ne pas demander un second guard sur un objet dont on tient déjà un guard en écriture** (ni `write` un objet qu'on est en train de lire dans le même thread) — le verrou d'un slot n'est pas réentrant. Les objets distincts, eux, se verrouillent librement, y compris imbriqués.

### API par fonctions de module

Rust :

```rust,ignore
use pyrucast::store::{insert, read, write, swap_out, compact};

let h = insert(mon_objet);              // dépose dans le store, renvoie un handle
let v = read(&h)?.valeur();             // guard lecture le temps de l'expression
write(&h)?.modifie();                   // guard écriture, pareil
swap_out(&h)?;                          // évince sur disque, libère la RAM
compact::<MonObjet>();                  // rétrécit la mémoire en queue de Vec
```

Python — configuration du répertoire de swap :

```python
import pyrucast
import pathlib

# Vérifier le répertoire de swap actuel (par défaut : répertoire temporaire OS).
print(pyrucast.swap_dir())

# Pointer vers un répertoire dédié (SSD NVMe, espace garanti).
pyrucast.set_swap_dir(pathlib.Path("/data/pyrucast_swap"))
print(pyrucast.swap_dir())  # /data/pyrucast_swap
```

> `swap_out` sur un objet individuel reste côté Rust pour l'instant. L'API Python exposera une fonction de haut niveau (Phase 5) pour déclencher l'éviction depuis un script.

### États du slot

```text
        ┌─────────────────────┐
        │      Resident       │  ◀──┐
        │   (valeur en RAM)   │     │  rechargement automatique
        └──────┬──────────────┘     │  au prochain read / write
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

### Fragmentation et compactage : ce qui borne la mémoire

Trois mécanismes coopèrent pour éviter que le `Vec` interne ne gonfle indéfiniment :

1. **La free-list** (vue plus haut). Les indices libérés sont **repris en priorité** par `insert` : si vous créez 1 000 objets puis en détruisez 999, le slot libéré sera réutilisé au prochain `insert` — pas d'extension du `Vec`. Le high-water mark reste borné par le nombre maximal d'objets vivants simultanément.

2. **`compact::<T>()`** — réduit le `Vec` quand sa **queue** est libre. Imaginez :

   ```text
   Avant compact :
   ┌───┬───┬───┬───┬───┬───┐
   │Res│Res│Free│Res│Free│Free│   capacité = 6
   └───┴───┴───┴───┴───┴───┘
                       ↑   ↑
                    queue libre

   Après compact::<T>() :
   ┌───┬───┬───┬───┐
   │Res│Res│Free│Res│            capacité = 4 (mémoire rendue à l'OS)
   └───┴───┴───┴───┘
   ```

   Le slot `Free` au milieu reste : `compact` **ne déplace pas** les slots vivants, pour ne pas invalider les handles existants. Conséquence : la fragmentation **interne** (trous au milieu du `Vec`) n'est pas résolue par cette opération — voir la table des évolutions plus bas (approche A).

3. **Le swap disque**. Quand la RAM devient un sujet plus pressant que le nombre de slots, `swap_out(&h)` sérialise la valeur vers un fichier (via `Persist`) et passe le slot dans l'état `OnDisk`. Le slot reste compté et adressable ; seule la valeur quitte la RAM. Le prochain `read` / `write` la recharge automatiquement.

Ces trois leviers se complètent : la free-list évite la croissance, `compact` rend la mémoire en queue, le swap déleste la RAM quand la queue est encombrée mais pas libérable.

### Concurrence : ce que le compilateur garantit

À travers un guard, **toute la sûreté est celle du borrow-checker Rust standard** : pas de mutation pendant qu'une lecture est active (le `RwLock` l'arbitre entre threads), pas de data race sur un slot. Le store n'invente aucune règle nouvelle — il fournit juste un **point d'entrée verrouillé** vers une référence Rust classique.

Une seule contrainte d'usage, non vérifiée à la compilation : **ne pas demander un second guard sur un objet pendant qu'on tient un guard en écriture sur ce même objet** (ni `write` un objet qu'on lit déjà dans le même thread) — le verrou du slot n'est pas réentrant. Les objets distincts — y compris du même type — se verrouillent librement et de façon imbriquée ; `insert`, `swap_out` et `compact` ne tiennent jamais de verrou de slot en même temps que celui du store.

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
