# Modèle mémoire

pyrucast gère la mémoire à deux niveaux, et deux seulement :

1. **Les objets** (`Coords`, `SubMesh`, `NodeField`, …) vivent derrière un `Handle<T>` — une référence comptée munie de son propre verrou. Le dernier handle qui disparaît emporte l'objet.
2. **Les nœuds**, à l'intérieur d'une `Coords`, portent leur propre compteur de références. Ils ne sont pas des objets : ce sont des indices dans un tableau, et c'est ce qui impose un second mécanisme.

Ce chapitre décrit les deux, et explique pourquoi le second ne se ramène pas au premier.

## Les objets : `Handle<T>`

### Ce que c'est

```rust,ignore
pub struct Handle<T> {
    cell: Arc<RwLock<T>>,
}
```

Une enveloppe d'un champ, et rien d'autre. `Arc` (*Atomically Reference Counted*) est la boîte partagée à tickets de propriété de la bibliothèque standard : la cloner ne copie pas le contenu, elle crée un second ticket sur la même boîte, libérée au dernier ticket rendu. `RwLock` est le verrou lecture/écriture qui arbitre les accès.

Trois propriétés en découlent, et ce sont les seules à retenir :

- **Compté.** `Clone` partage, `Drop` relâche. Quand le dernier handle disparaît, la valeur est détruite — son `Drop` s'exécute, donc les effets de bord (un `SubMesh` qui rend ses nœuds à la `Coords`) ont lieu **exactement une fois**.
- **Toujours valide.** Un handle ne peut pas survivre à son objet : le détenir, c'est le maintenir en vie. Il n'y a pas de handle périmé, pas de génération à vérifier, et `read` / `write` ne peuvent pas échouer.
- **L'identité, c'est le pointeur.** `same_object` répond à « ces deux références désignent-elles le même objet ? » — c'est ce qui permet à l'union des agrégats d'ignorer une zone qu'elle possède déjà, et à un champ de reconnaître que son support est bien le maillage qu'on lui a passé.

```rust,ignore
let h1 = Handle::new(MaStruct(42));   // premier ticket
let h2 = h1.clone();                  // même objet
assert!(h1.same_object(&h2));
drop(h1);                             // h2 tient encore l'objet
drop(h2);                             // dernier ticket → l'objet est détruit
```

Aucun `remove()` à appeler, aucune fonction de libération : la portée Rust suffit.

### Les guards : lire et écrire en place

Avec un handle, on n'écrit pas `handle.dim` directement. On passe par un **guard** :

```rust,ignore
let coords = handle.read();             // verrou lecture sur CET objet seul
println!("dim = {}", coords.dim());     // coords se comporte comme un &Coords
drop(coords);                           // (ou fin de portée) → verrou relâché

handle.write().add_node(&[0.0, 0.0])?;  // verrou écriture, le temps de l'appel
```

Le guard prouve qu'on détient le verrou. Il se comporte comme une référence vers la donnée (par `Deref`), et **le verrou est relâché à sa destruction** — c'est du RAII : impossible d'oublier de déverrouiller. Les règles du borrow-checker s'appliquent à travers lui, arbitrées à l'exécution par le `RwLock` : N lecteurs simultanés, ou 1 écrivain exclusif.

Un point compte pour les performances : les guards sont **possédés** (`'static`). Ils détiennent leur propre ticket sur l'objet, donc ils peuvent être renvoyés par une fonction, rangés dans une struct — c'est le mécanisme de `FieldView`, la vue zéro-copie des champs — et ils maintiennent l'objet en vie même si le handle dont ils viennent disparaît entre-temps. C'est ce qui permet aux opérateurs (gradient, solveur, visualisation) de lire les données **en place** pendant toute leur boucle, au lieu d'en faire des copies.

### Concurrence : un verrou par objet

Il n'y a pas d'annuaire global, donc rien à sérialiser entre threads en dehors des objets eux-mêmes :

- deux threads qui manipulent **des objets différents** — même du même type — ne se gênent jamais ;
- plusieurs threads peuvent **lire le même objet** simultanément ;
- un écrivain est exclusif sur **son** objet, et seulement le sien.

C'est la granularité dont vit le parallélisme interne aux opérateurs : plusieurs threads lisent le même maillage pendant l'assemblage (voir [Parallélisme](developper/parallelisme.md)).

**Une seule règle d'usage**, non vérifiée à la compilation : **ne pas demander un second guard sur un objet dont on tient déjà un guard en écriture** (ni `write` un objet qu'on est en train de lire dans le même thread) — le `RwLock` n'est pas réentrant. Les objets distincts se verrouillent librement, y compris de façon imbriquée.

### Affichage : `<SubMesh #7f3a2c>`

Formater un handle n'ouvre **pas** l'objet :

```text
<SubMesh #7f3a2c>
```

Le nombre est l'adresse de l'objet, tronquée. Il sert à vérifier à l'œil, dans une trace, que deux lignes parlent bien du même objet. Ce n'est **pas** une identité pérenne : une adresse est réutilisée une fois l'objet libéré, donc deux entrées portant le même repère dans un journal écrit au fil du temps peuvent désigner deux objets différents.

Le choix de ne rien lire est délibéré : un `SubMesh` porte une connectivité de plusieurs millions d'entrées, et un handle peut fort bien être formaté alors qu'un guard en écriture est tenu sur lui — le lire bloquerait.

### Pourquoi un `Handle` plutôt qu'un `Arc<RwLock<T>>` nu ?

L'enveloppe ne coûte rien à l'exécution et rapporte trois choses :

1. **Des méthodes propres** — `h.read()` plutôt qu'un trait d'extension importé partout.
2. **Une surface réduite** — un `Arc` nu exposerait `try_unwrap`, `get_mut`, `strong_count` : de quoi contourner le verrou.
3. **Un entonnoir de création unique** — `Handle::new`. Si l'énumération des objets vivants devient un jour souhaitable (un listing à la cast3m, une sauvegarde de session entière), elle coûte un `Vec<Weak<_>>` inscrit dans ce seul constructeur, au lieu d'une chasse à travers tous les sites de création.

### Ce qui a précédé

Il y a eu un store global : un registre par `TypeId`, un tableau de cases numérotées et générationnelles, une free-list, un compteur de références écrit à la main, et un délestage sur disque. Chaque pièce doublait quelque chose que l'`Arc` faisait déjà :

- le compteur maison suivait ce que suivait celui de l'`Arc` ;
- la génération ne protégeait que les handles reconstruits depuis des octets, que seul le délestage produisait ;
- et ce délestage, lui, ne libérait rien : il écrivait sur disque puis oubliait la valeur au lieu de la détruire.

Ce que le registre offrait en plus — énumérer les objets vivants, leur donner un numéro stable — n'a jamais atteint un utilisateur. Il a donc été retiré ; l'entonnoir `Handle::new` garde la porte ouverte si le besoin se présente.

## Les nœuds : un compteur dans la `Coords`

À l'intérieur d'une `Coords`, chaque nœud porte son propre compteur. Il est incrémenté quand un `Node` est cloné ou qu'un `SubMesh` référence le nœud dans une maille, décrémenté quand ils disparaissent ; `gc()` collecte les nœuds retombés à zéro. Le détail est au chapitre [Coords](coords.md).

**Pourquoi ne pas leur donner un `Handle` comme aux objets ?** Parce qu'un nœud n'est pas un objet : c'est un indice dans un tableau de connectivité. Un `SubMesh` de 100 k HEX20 en porte deux millions d'occurrences ; en faire autant de pointeurs comptés multiplierait la mémoire de la connectivité et détruirait sa localité. Le compteur reste donc un `Vec<u32>` parallèle au tableau de coordonnées, où le nœud n'occupe que quatre octets de comptage.

### Pourquoi un compteur plutôt qu'un mark-and-sweep ?

L'alternative serait un GC **mark-and-sweep** : pas de compteur, mais à chaque `gc()`, parcourir tous les `Node` / `SubMesh` / `NodeField` vivants pour marquer les `NodeId` atteignables. La simplification est apparente — il faut d'abord savoir **où sont les racines**.

Un `SubMesh` ou un `NodeField` est atteignable depuis les handles que l'utilisateur tient. Mais un `Node` vit **sur la pile Rust** (ou dans le tas Python via PyO3), et rien ne l'énumère. Deux options seulement :

1. Maintenir dans la `Coords` une **liste séparée** des `Node` vivants, mise à jour à `Clone` / `Drop`. C'est le compteur actuel déguisé en `HashSet<NodeId>` — sans gain.
2. **Changer le contrat** : `Node` devient une vue non protectrice, et seul un maillage peut maintenir un nœud vivant. C'est exactement le modèle cast3m (« un point isolé n'existe pas en dehors d'un MAILLAGE »). Légitime, mais c'est une rupture d'API plus large que la simplification cherchée.

Où est le coût réel du compteur ?

- `SubMesh::add_cell` : `N × incref` sous **un seul** verrou — négligeable devant la création de la maille.
- `Node::clone` / `drop` : un verrou et une indexation de `Vec<u32>`. Sensible *seulement* si on clone ou détruit beaucoup de `Node` en boucle serrée, ce qui est rare en calcul EF — on construit le maillage, puis on n'y touche plus.
- **Complexité conceptuelle** : la logique d'annulation dans `add_cell` pèse plus sur la lisibilité que sur le temps de calcul.

**Choix pyrucast** : compteur pour l'instant. Si la complexité devient gênante, le levier le plus rentable n'est pas de remplacer le mécanisme mais de **simplifier le contrat de `Node`** (option 2, mode cast3m pur) — cela supprime à la fois le compteur *et* la logique d'annulation. À garder en tête si l'on observe que les `Node` isolés servent peu en pratique.

## Sauvegarde sur disque

Le trait `Persist` (`serde` + `bincode`) est l'unique mécanisme de sérialisation, au format binaire identique Linux ↔ Windows (voir [Conventions](conventions.md)).

La sauvegarde d'un **graphe d'objets** — écrire plusieurs objets dans un fichier et les relire en préservant leur partage, de sorte que deux champs portés par le même support restent, après relecture, deux champs portés par un seul support — est **décidée mais pas encore écrite**. Elle reposera sur des racines nommées, des enregistrements explicites par type munis d'identifiants locaux au fichier, et des compteurs jamais sauvegardés mais recomptés à la relecture. Voir la [feuille de route](https://github.com/Pyrucast/pyrucast/blob/master/ROADMAP.md).
