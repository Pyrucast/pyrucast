# Maillage quadrangulaire — état de l'art et pistes

Note de travail, **non arbitrée**. Elle dit où se situent les mailleurs
quadrangulaires de pyrucast dans la littérature, ce qui les sépare de l'état de
l'art, et les six pistes identifiées — avec, pour chacune, ce qui a été mesuré.

État arrêté au **24 août 2026**. Compagnon de [ROADMAP.md](ROADMAP.md), section
*Qualité de maillage*.

La **piste 7** a été ouverte et fermée le 24 août : elle est implémentée, et
c'est le seul point de cette note qui ne soit plus une piste.

---

## 1. D'où vient cette note

Une boîte crénelée à treize plaques, maillée par `pave_surface` et par
`grid_surface`, a servi de cas d'étude. Elle a fait sortir quatre défauts, tous
corrigés :

| commit | défaut |
|---|---|
| `6bf4f6f` | une grille de voisinage plate allouait 4 Gio et tuait le processus |
| `e53785a` | la couture d'un anneau plat laissait une fissure, ouverte en trou par le lissage |
| `41594a9` | une couture n'emportait pas les triangles, et pouvait poser une corde sur le contour |
| `3c78950` | la pile LIFO des boucles écrasait la première rangée d'un contour |
| `946677c`, `51afb9f` | deux gestes de nettoyage : paires de triangles plats, et nœud partagé par un triangle et deux quadrangles |

Après quoi, sur ce cas : **pire maille 0,451**, 3 mailles sous 0,5 sur 10 120,
zéro trou. Ce qui reste est structurel, d'où cette note.

---

## 2. Les cinq lignées de la littérature

### 2.1 Le pavage direct — la lignée de `pave_surface`

Blacker & Stephenson, *Paving: a new approach to automated quadrilateral mesh
generation*, IJNME **32**:811–847 (1991). Un front qui avance en rangées, avec
couture, déblocage et fermeture.

Toujours travaillée : *[An improved Q-Morph algorithm for quad-dominant hybrid
mesh generation with advanced front propagation and topology
optimization](https://link.springer.com/article/10.1007/s00366-025-02196-y)*,
*Engineering with Computers* **41**:4255–4275 (2025), enrichit les types de
front pour les concavités et ajoute une optimisation topologique par **gabarits
prédéfinis, remaillage de cavités et élimination des paires de triangles** —
c'est-à-dire exactement les gestes ajoutés dans `946677c` et `51afb9f`.

### 2.2 L'indirect frontal — Q-Morph

Owen, Staten, Canann & Saigal, *[Q-Morph: an indirect approach to advancing
front quad
meshing](https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-0207(19990330)44:9%3C1317::AID-NME532%3E3.0.CO;2-N)*,
IJNME **44**:1317–1340 (1999). Trianguler d'abord, puis transformer les
triangles dans un ordre dicté par un front.

**Dépassé** par la lignée suivante dès 2012 : à ne pas écrire aujourd'hui.

### 2.3 L'indirect optimal — Blossom

Remacle, Lambrechts, Seny, Marchandise, Johnen & Geuzaine, *[Blossom-Quad: a
non-uniform quadrilateral mesh generator using a minimum-cost perfect-matching
algorithm](https://onlinelibrary.wiley.com/doi/10.1002/nme.3279)*, IJNME
**89**:1102–1119 (2012).

Le graphe a un sommet par triangle, une arête par paire adjacente, pondérée par
la qualité du quadrangle que la paire ferait. On y résout le **couplage parfait
de coût minimal** (algorithme d'Edmonds), en temps polynomial et **exactement** :
aucune passe locale ne peut faire mieux, et si le compte de triangles est pair
et le couplage parfait existe, il ne reste aucun triangle.

Complété par le mailleur triangulaire fait pour ça : Remacle et al., *[A frontal
Delaunay quad mesh generator using the L∞
norm](https://onlinelibrary.wiley.com/doi/10.1002/nme.4458)*, IJNME (2013), qui
produit des triangles **presque rectangles** — la norme L∞ est ce qui fait que
les paires donnent des carrés.

### 2.4 Champs de directions et paramétrisation — la ligne dominante

On ne construit pas des mailles, on construit un **champ**, et le maillage en
tombe. Un champ de croix (directions à symétrie d'ordre 4) porte des
**singularités** qui sont exactement les futurs sommets irréguliers, et
Poincaré–Hopf impose la somme de leurs indices : **le nombre d'irréguliers n'est
pas négociable, seule leur position l'est.**

Points d'entrée obligés — les deux revues :

- Bommes, Lévy, Pietroni, Puppo, Silva, Tarini & Zorin, *[Quad-Mesh Generation
  and Processing: A Survey](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.12014)*,
  CGF **32**:51–76 (2013) ;
- Campen, *Partitioning Surfaces into Quadrilateral Patches: A Survey*, CGF
  **36**:567–588 (2017).

Jalons : QuadCover (Kälberer et al., 2007), *Mixed-Integer Quadrangulation*
(Bommes et al., 2009), *[Globally optimal direction
fields](https://dl.acm.org/doi/10.1145/2461912.2462005)* (Knöppel et al., 2013),
*[Integrable PolyVector fields](https://dl.acm.org/doi/10.1145/2766906)*
(Diamanti et al., 2015).

La **quantification** — passer d'un champ continu à un maillage à arêtes
entières : *Quantized global parametrization* (Campen et al., 2015), *Quad
layouts via constrained T-mesh quantization* (Lyon et al., 2021),
*Min-deviation-flow in bi-directed graphs for T-mesh quantization* (Heistermann
et al., 2023). Et la théorie : *Which cross fields can be quadrangulated?* (Shen
et al., 2022) — tous les champs ne le sont pas.

Le côté rapide et pragmatique : *[Instant Field-Aligned
Meshes](https://dl.acm.org/doi/10.1145/2816795.2818078)* (Jakob, Tarini, Panozzo
& Sorkine-Hornung, SIGGRAPH Asia 2015) — lissage local conjoint d'un champ
d'orientation et d'un champ de position, sans optimisation globale, donc
linéaire et interactif.

Le plus récent et le plus pertinent ici : Couplet, Chemin, Bommes & Chien,
*[Surface Quadrilateral Meshing from Integrable Odeco
Fields](https://arxiv.org/abs/2604.03889)* et *Size-controlled quadrilateral
meshing using integrable odeco fields*, SGP 2026 — la **carte de tailles** faite
proprement, par champs de repères intégrables à contraintes d'alignement **et de
taille**.

### 2.5 L'école de la grille — la lignée de `grid_surface`

Liang & Zhang, *[Guaranteed-quality all-quadrilateral mesh generation with
feature
preservation](https://www.sciencedirect.com/science/article/abs/pii/S0045782510000836)*,
CMAME (2010), et *Hexagon-based all-quadrilateral mesh generation with
guaranteed angle bounds*, CMAME (2011).

Quadtree gouverné par la courbure, **gabarits 2-raffinement sans nœud pendant**,
puis une **zone tampon** de deux couches créée en retirant les éléments près du
bord. Garantie dure : **tous les angles dans [45°, 135°]**.

C'est structurellement `grid_surface` (cœur en grille + bande frontale), à deux
différences près : eux graduent par quadtree — ce que `grid_surface` a essayé
puis retiré — et surtout ils **prouvent** une borne d'angle au lieu de la
mesurer. Plus récent : *[Boundary constrained quadrilateral mesh generation
based on domain decomposition and
templates](https://www.sciencedirect.com/science/article/abs/pii/S004579492400004X)*,
*Computers & Structures* (2024).

### 2.6 La vague neuronale (2024–2026) — à situer

*Learning Direction Fields for Quad Mesh Generation* (Dielen et al., 2021),
*[NeurCross](https://dl.acm.org/doi/10.1145/3731159)* (ACM TOG 2025), puis les
autorégressifs — *[QuadGPT](https://arxiv.org/html/2509.21420v1)*, QuadLink,
TopGen (2026). Ils visent la rétopologie « game-ready », **pas le calcul** :
aucune garantie de validité ni de respect exact du contour. Bibliographie
vivante : [quad-meshing-survey](https://github.com/Bigger-and-Stronger/quad-meshing-survey).

---

## 3. La carte : pyrucast ↔ littérature

| ce que fait pyrucast | l'état de l'art correspondant | l'écart |
|---|---|---|
| `pave_surface` | Paving 1991, + Q-Morph amélioré 2025 | **à jour sur cette branche** : les trois gestes de l'article 2025 sont ceux de `946677c` / `51afb9f` |
| `grid_surface`, `grid_surface2` | Liang & Zhang 2010–2011 | même architecture (cœur + tampon) ; la **garantie d'angle** en moins |
| `merge_triangles` (glouton : fusion + regroupement) | Blossom-Quad 2012 | **une génération de retard sur un problème identique** |
| `cleanup` (doublets, valences, étoiles à trois mailles) | CleanUp 1997 ; QuadQS : cavités + motifs guidés par les singularités | **à jour sur le geste de base** (piste 7) ; reste qu'on répare à l'aveugle, quand eux savent **où** un irrégulier a le droit d'être |
| taille : un scalaire par domaine | odeco intégrables, SGP 2026 | absent |
| — | champ de croix, quantification, layout | absent, et c'est la colonne vertébrale du reste |

Gmsh est la synthèse de tout cela : Reberol, Georgiadis & Geuzaine,
*[Quasi-structured quadrilateral meshing in
Gmsh](https://arxiv.org/abs/2103.04652)* (2021) — champ de croix + carte de
tailles → insertion frontale → recombinaison Blossom → **subdivision par les
milieux** (tout-quadrangle garanti) → remaillage topologique gardant les
irréguliers qui correspondent à une singularité du champ, chaque opération
annulée si la qualité baisse.

---

## 4. Les six pistes

### Piste 1 — Le couplage optimal dans `merge_triangles`

Remplacer la passe gloutonne (fusion, puis regroupement en hexagone) par le
**couplage parfait de coût minimal** de Blossom-Quad.

- *Mesuré* : sur un cadre crénelé, `triangulate_surface` en QUA4 laisse
  **147 triangles sur 588 mailles**, pire maille 0,000 (0,331 après
  `regularize` + `cleanup`). Un couplage exact en laisserait zéro ou un.
- *Réserve* : l'implémentation de référence (Blossom V, Kolmogorov) est sous
  licence recherche, **incompatible MPL-2.0**. Il faut écrire Edmonds ou trouver
  une caisse permissive.
- *Verdict* : **le meilleur rapport gain/effort de la liste.** Problème connu,
  solution exacte publiée, périmètre borné à un opérateur.

### Piste 2 — La discipline de parité, hors `all_quad`

Les deux endroits qui décident d'un écart de découpe imposent déjà « les deux
moitiés restent paires », mais **seulement sous `all_quad`** :

```rust
// unstick
if gap < 3 || n - gap < 3 || (all_quad && gap.is_multiple_of(2)) { continue; }
// find_seam
if all_quad && gap % 2 == 1 { continue; }
```

Or les 65 triangles de la boîte viennent **tous** de la fermeture d'un petit
anneau (49 d'un anneau à 3 nœuds), et la parité d'un anneau se décide dans les
**découpes**, pas dans les rangées.

- *Mesuré*, discipline activée seule (sans exiger le zéro triangle) :

  | | actuel | parité active |
  |---|---:|---:|
  | boîte, pave `round` `along` | 65 tri, pire 0,451, 5ᵉ c. 0,551 | **27** tri, 0,423, 0,451 |
  | boîte, pave `round` `none` | 133 tri, 0,257 | **75**, **0,343** |
  | boîte, `grid_surface` | 65 tri, 5ᵉ c. 0,476 | **9** tri, 5ᵉ c. **0,550** |

- *Point ouvert* : coûte +1 263 mailles sur la configuration `round` + `along`
  et y fait redescendre le 5ᵉ centile de 0,551 à 0,451. Comprendre pourquoi là
  et pas ailleurs.
- *Verdict* : **une journée, les chiffres sont déjà là.**

### Piste 3 — La subdivision par les milieux

Filet tout-quadrangle inconditionnel : chaque triangle donne trois quadrangles,
chaque quadrangle quatre. ×4 mailles, taille divisée par deux, zéro triangle,
**et jamais un refus de contour**. C'est ce que fait gmsh dans QuadQS avant le
remaillage topologique.

- *Motivation directe* : `all_quad=True` **refuse** aujourd'hui un contour de
  parité impaire (« the outer boundary loop has 1151 segments — an odd
  number »). La boîte de l'étude n'y a donc pas droit.
- *Verdict* : quelques dizaines de lignes, aucun risque, mais un compromis de
  densité que l'appelant doit choisir.

### Piste 4 — Une carte de tailles

La taille visée est un **scalaire par domaine**. Les mailleurs de l'état de
l'art acceptent un champ de tailles avec limite de gradient ; `grid_surface`
prend bien ses lignes sur le contour, mais l'appelant ne peut pas dicter une
densité variable.

- *Verdict* : **le seul point de la liste que les utilisateurs verraient
  directement.** Une semaine environ pour une version scalaire interpolée sur un
  maillage de fond ; la version « propre » demande la piste 6.

### Piste 5 — L'atterrissage sur un front vivant (`aim_at_live`)

`aim_at_frozen` raccourcit déjà l'avance pour **poser** une rangée sur un cœur
en grille. Rien d'équivalent n'existe entre deux fronts vivants : ils se
percutent.

- *Verdict* : **l'intérêt a fondu.** C'était la réponse à l'écrasement du front,
  réglé autrement par `3c78950` et `51afb9f` — il ne reste que 3 mailles sous
  0,5 sur 10 120. À garder en réserve, pas en priorité.

### Piste 6 — Le champ de croix

Le vrai saut, et la condition des autres : il dit **où** un sommet irrégulier a
le droit d'être, il porte la carte de tailles, et il ouvre le layout.

- *Verdict* : **un projet**, et le seul de la liste qui demande de la théorie
  autant que du code. C'est ce qui séparerait pyrucast de l'état de l'art plutôt
  que de l'état de l'art de 1991.

### Piste 7 — L'effondrement des étoiles pauvres — **FAITE**

Un nœud intérieur qui n'a que **trois** mailles autour de lui peut être
abandonné, et une maille avec lui. Kinney, *[CleanUp: Improving Quadrilateral
Finite Element Meshes](https://people.eecs.berkeley.edu/~jrs/meshpapers/Kinney.pdf)*,
4ᵉ IMR (1997), cas `3-4+34+000` : « *all three quads around the center node are
deleted and a fill_2 is used to fill the hole. Four irregular nodes are replaced
with zero irregular nodes.* »

`cleanup` en avait **un** des quatre cas, écrit comme un cas particulier
(« pentagone », deux quadrangles et un triangle). L'identité qui les unifie :
autour d'un nœud portant \( q \) quadrangles et \( t \) triangles, chaque
quadrangle pose deux arêtes qui ne le touchent pas et chaque triangle une, donc
l'étoile est bordée par un polygone à \( n = 2q + t \) côtés ; et une
décomposition d'un \( n \)-gone sans nœud intérieur vérifie
\( 2q' + t' = n - 2 \). Avec \( q + t = 3 \), la redécoupe existe toujours,
et toujours avec une maille de moins.

| \( q, t \) | bord | avant | après | mailles |
|---|---|---|---|---|
| 3, 0 | hexagone | 3 quadrangles | 2 quadrangles | 3 → 2 |
| 2, 1 | pentagone | 2 quadrangles, 1 triangle | 1 de chaque | 3 → 2 |
| 1, 2 | quadrangle | 1 quadrangle, 2 triangles | 1 quadrangle | 3 → 1 |
| 0, 3 | triangle | 3 triangles | 1 triangle | 3 → 1 |

**Le point qui a coûté trois essais** : le geste ne peut pas être jugé sur
place. Il *supprime* un nœud, donc l'anneau restant est mécaniquement étiré
jusqu'à ce que quelque chose le relâche — ce qui arrive toujours, les paveurs
lissant après chaque rangée. Mesurée sur-le-champ, la redécoupe paraît presque
toujours moins bonne que l'étoile qu'elle remplace : le plancher de qualité de
`switch_diagonals` (70 %), appliqué tel quel, supprimait **53 gestes utiles sur
61**. Or `switch_diagonals` peut se le permettre — il ne déplace aucun nœud,
donc ce qu'il mesure est définitif.

L'ordre retenu, qui est celui de gmsh : appliquer, **relaxer l'anneau**,
mesurer, défaire entièrement si la pire maille du voisinage a baissé. Trois
détails s'y sont révélés décisifs, chacun par une mesure :

- la relaxation doit être **gardée** avec le geste. Juger sur des positions
  qu'on rejette ensuite, c'est mesurer un maillage que personne ne reçoit ;
- la relaxation d'essai doit porter **la même garde que le vrai lisseur** (pas
  de pas qui retourne une maille). Un laplacien nu marche, près d'un coin
  concave, vers un point que le lisseur monotone n'atteindra jamais, et fait
  accepter le geste sur une promesse qui ne sera pas tenue : la maison tombait
  à **0,055** de pire maille ;
- un **pré-filtre** garde l'entrée : rien n'est tenté si le geste n'apporte
  ni valence ni forme. Le verdict après relaxation juge le *voisinage*, donc
  localement, et ne voit pas qu'une maille médiocre ailleurs vient de devenir
  la pire du maillage. Sans lui on gagne treize irréguliers et on perd la
  garantie de non-régression sur la pire maille — l'échange est mauvais, une
  pire maille qui recule casse un calcul.

**La paire 3-3, ajoutée ensuite.** Deux nœuds intérieurs de valence 3 reliés
par une arête sont hors de portée du geste ci-dessus : abandonner l'un seul
échange un irrégulier contre deux, et le pré-filtre le refuse. Ensemble ils ne
portent que **quatre** quadrangles — leurs étoiles se recouvrent sur les deux
mailles de l'arête commune — bordés par un **hexagone** dans les vingt-sept cas
trouvés sur la boîte, sans exception. Deux nœuds et deux mailles partent d'un
coup, pour un gain de valence de +2, jusqu'à +4 quand l'anneau porte un 5.
Examinée **avant** le nœud seul, faute de quoi celui-ci prend l'un des deux et
la paire n'a jamais sa chance.

Étendue ensuite aux paires **3-4**, dont l'étoile porte cinq mailles bordées
par un **heptagone**, recoupé en deux quadrangles et un triangle — celui qui
était déjà là, la parité interdisant d'en créer un.

| | `grid_surface` | `pave_surface` |
|---|---:|---:|
| irréguliers | 185 → 164 → **104** | 739 → **703** → 708 |
| erreur de valence | 194 → 164 → **104** | 754 → **718** → 714 |
| valence 3 internes | 58 → 46 → **16** | 329 → **311** → 312 |
| 1ᵉʳ centile | 0,706 → 0,714 → **0,824** | 0,625 → 0,633 → **0,641** |
| pire maille | 0,461 → 0,461 → 0,437 | 0,141 → **0,284** → 0,284 |

*(colonnes : avant la paire, paire 3-3, puis 3-4.)* Sur `grid_surface`, deux
paires 3-3 détectées font tomber douze nœuds de valence 3 et non quatre : un
effondrement en débloque d'autres par cascade. Le cas 3-4 en enlève trente de
plus, au prix de 0,019 sur la pire maille — arbitrage assumé, le premier
centile gagnant 0,11 dans le même mouvement.

*Mesuré*, sortie brute des paveurs, avant → après :

| | mailles | pire | 1ᵉʳ c. | 5ᵉ c. | irréguliers | erreur de valence |
|---|---:|---:|---:|---:|---:|---:|
| boîte, `grid_surface` | 11 749 → **11 523** | 0,456 → **0,461** | 0,572 → **0,706** | 0,760 → **0,844** | 568 → **185** | 724 → **194** |
| cercle, `grid_surface` | 1 260 → **1 236** | 0,288 → **0,366** | | | | |
| maison, `grid_surface` | 470 → **460** | 0,420 → 0,420 | | | | |
| carré arrondi, `grid` | 441 → **424** | 0,244 → **0,340** | | | | |
| carré arrondi, `grid2` | 415 → **409** | 0,400 → 0,371 | | | | |

Sur la boîte, les **274** nœuds intérieurs de valence 3 étaient *tous* entourés
de trois quadrangles — le seul cas que `cleanup` ne savait pas traiter. Il en
reste 58, et c'est le plancher : Poincaré–Hopf fixe le nombre d'irréguliers,
seule leur position est négociable (§ 2.4).

Le seul retrait est le carré arrondi en `grid_surface2`, 0,400 → 0,371.

**Un bug indépendant, trouvé au passage.** Les mesures de qualité sont signées,
et une maille lue en sens horaire compte négatif — ce qui se lit comme
*retournée*. Un maillage entièrement horaire n'a pourtant rien d'anormal : un
paveur rend le sens du contour qu'on lui a donné, si bien qu'un domaine maillé
depuis un contour extérieur inversé sort horaire. Lu tel quel, **les trois
opérateurs `improve` refusaient silencieusement de le toucher** : `regularize`
ne déplaçait aucun nœud (déplacement maximal mesuré : `0.0`), `cleanup` ne
trouvait rien, et `merge_triangles` laissait les 74 triangles de la boîte à 74
— contre 64 une fois le maillage retourné. Corrigé dans `Surface::read`, qui
normalise le sens à la lecture et le restitue à la sortie.

---

## 5. Ce qui a été essayé et écarté

À ne pas refaire sans raison nouvelle. Tous ces essais ont été mesurés puis
retirés du dépôt.

| essai | résultat |
|---|---|
| Abandonner la rangée dès qu'**un** nœud passe sous le plancher de détente (seuils 0,5 / 0,35 / 0,25 / 0,15) | une maille à jacobien **0,000** apparaît à chaque seuil, y compris sur le cercle en grille |
| Démotion en fin de rangée d'un nœud qui ne peut plus avancer | gain net sur l'anguleux (carré 20×20 : 600 → 544 mailles, pire 0,541 → 0,778) mais **maison 0,491 → 0,147** et **cercle en grille 5ᵉ c. 0,796 → 0,608** |
| Parcours **FIFO** complet des boucles | ta boîte gagne, mais **`grid_surface` + `along` : pire 0,312 → 0,047** |
| Parcours FIFO **pour toutes** les boucles, y compris sous cœur en grille | `grid_surface` sur une plaque à trou rond **ne termine plus** (> 4 min contre 0,05 s) |
| Forcer un front pair à **chaque** rangée | 65 triangles → **685**, pire maille 0,000 |
| Idem, restreint aux dernières rangées avant fermeture (seuils 8 / 12 / 20 / 40) | aucun gain ; à 12, `grid_surface` refait des trous |
| Plancher de qualité **immédiat** sur l'effondrement d'une étoile (70 %, celui de `switch_diagonals`) | 53 gestes utiles sur 61 supprimés. Le geste retire un nœud : il ne se juge qu'après relaxation (piste 7) |
| Relaxation d'essai **rendue** après le verdict | mesure un maillage que personne ne reçoit : carré arrondi `grid2` 0,468 en essai, 0,331 livré |
| Relaxation d'essai en **laplacien nu**, sans garde de validité | promet une position que le lisseur monotone n'atteint pas : maison `grid_surface` pire maille **0,055** |
| Effondrement **sans pré-filtre** : le verdict après relaxation pour seule condition | gagne sur tout ce qu'on visait — boîte 185 → **162** irréguliers, 58 → **45** nœuds de valence 3, 11 523 → 11 506 mailles, et `pave_surface` en profite aussi (maison 620 → 595) — mais la pire maille **recule sous le point de départ** : boîte 0,461 → 0,436 pour 0,456 au départ, maison `grid_surface` 0,420 → **0,346**. Le verdict est local et ne voit pas qu'une maille médiocre ailleurs est devenue la pire du maillage. Rendre le verdict global est un autre chantier |
| Refuser la couture qui poserait une corde sur le contour | règle bien les trous, mais sur une bande crénelée le front se replie et **l'appel échoue** — perdre le maillage pour éviter une fissure d'aire nulle est un mauvais échange. Remplacé par la recouture de `41594a9` |

Un point de méthode qui a servi plusieurs fois : **la relaxation du front et
l'ordre de parcours gagnent sur les formes anguleuses et perdent sur les
courbes.** Tout réglage global de ces deux leviers se paie quelque part ; c'est
pourquoi la relaxation est devenue un choix de l'appelant (`relax`) plutôt qu'un
réglage imposé.
