# Maillage quadrangulaire — état de l'art et pistes

Note de travail, **non arbitrée**. Elle dit où se situent les mailleurs
quadrangulaires de pyrucast dans la littérature, ce qui les sépare de l'état de
l'art, et les six pistes identifiées — avec, pour chacune, ce qui a été mesuré.

État arrêté au **23 août 2026**. Compagnon de [ROADMAP.md](ROADMAP.md), section
*Qualité de maillage*.

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
| `cleanup` (doublets, valences, pentagone) | QuadQS : cavités + motifs guidés par les singularités | on répare localement à l'aveugle ; eux savent **où** un irrégulier a le droit d'être |
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
| Refuser la couture qui poserait une corde sur le contour | règle bien les trous, mais sur une bande crénelée le front se replie et **l'appel échoue** — perdre le maillage pour éviter une fissure d'aire nulle est un mauvais échange. Remplacé par la recouture de `41594a9` |

Un point de méthode qui a servi plusieurs fois : **la relaxation du front et
l'ordre de parcours gagnent sur les formes anguleuses et perdent sur les
courbes.** Tout réglage global de ces deux leviers se paie quelque part ; c'est
pourquoi la relaxation est devenue un choix de l'appelant (`relax`) plutôt qu'un
réglage imposé.
