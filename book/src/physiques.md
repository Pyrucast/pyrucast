# Détails des physiques

Chaque **physique** est une variante de [`SubModel`](model.md) : elle déclare
ses variables (primales / duales), son matériau, et sait assembler sa rigidité
`K` (et, le cas échéant, intégrer son comportement `COMP`). Le `Model`
orchestre, mais ne porte aucune logique physique — voir
[Modèle physique](model.md) pour la mécanique générique (DOFs, assemblage,
chargements, solveur).

Chaque page suit le **même plan standard**, dans cet ordre :

1. **Introduction** — nature du modèle, éléments et degrés de liberté.
2. **Équations continues résolues** — forme forte (et faible) du problème.
3. **Forme discrétisée** — opérateurs discrets (`B`, `N`) et expressions des
   matrices élémentaires (`K`, masse/capacité, tangente…).
4. **Variables et matériau** — noms primal/dual, composantes matériau,
   comportement (`COMP`) et, le cas échéant, état interne.
5. **Mise en donnée (Rust, testé)** — exemple Rust exécuté via `{{#include}}`.
6. **Exemple Python** — l'équivalent haut niveau.
7. **Compléments** (facultatif) — extensions propres au modèle (masse &
   rigidité géométrique, thermomécanique, convection, régime transitoire…).

Certaines physiques omettent une rubrique sans objet (p. ex. pas d'exemple Rust
dédié pour un modèle piloté par le comportement) ; l'ordre reste identique.

- [Conduction thermique](thermique.md) — `-∇·(k∇T) = 0`, l'exemple canonique,
  la [conduction orientée](thermique.md#conduction-orthotrope-et-anisotrope)
  (tenseur `K`) et la
  [convection de surface](thermique.md#convection-de-surface-robin--film)
  (Robin / film, `q·n = h(T − T_ext)`), plus le
  [rayonnement à l'infini](thermique.md#rayonnement-à-linfini-stefan-boltzmann)
  (`q·n = σε(T⁴ − T_∞⁴)`, non linéaire, à tangente cohérente).
- [Diffusion (loi de Fick)](diffusion.md) — `∇·(D∇c) = 0`, concentration `c` et
  flux de matière `j` ; même opérateur que la conduction, nature distincte. Avec
  le [transfert d'interface](diffusion.md#transfert-à-travers-une-interface)
  `j·n = h(c₁ − c₂)`, qui laisse le champ sauter entre deux corps.
- [Mécanique](mecanique.md) — barre, élasticité linéaire, poutres et
  portiques :
  - [Barre / treillis](mecanique/truss.md)
  - [Élasticité linéaire](mecanique/elasticite.md)
  - [Élasticité orthotrope et anisotrope](mecanique/orthotropie.md)
  - [Pression suiveuse](mecanique/pression-suiveuse.md)
  - [Plasticité parfaite (von Mises)](mecanique/plasticite.md)
  - [Lois d'écoulement plastique](mecanique/lois-plastiques.md) — écrouissage
    isotrope, Drucker-Prager, Ottosen
  - [Fluage et viscoplasticité](mecanique/fluage.md) — Norton, Lemaitre,
    Blackburn, Chaboche et sa variante endommageable
  - [Endommagement de Mazars](mecanique/mazars.md)
  - [Lois d'endommagement](mecanique/endommagement.md) — Damage TC, SiC/SiC
    orthotrope, plasticité poreuse de Gurson
  - [Poutre d'Euler-Bernoulli](mecanique/bernoulli.md) — 1-D, plan, spatial ;
    exacte aux nœuds
  - [Poutre de Timoshenko](mecanique/timoshenko.md)
  - [Portique 2D](mecanique/timoshenko.md)
  - [Cadre 3D](mecanique/timoshenko.md)
  - [Coques](mecanique/coques.md) — Reissner-Mindlin, six DDL par nœud
- [Contraintes](contraintes.md) — conditions limites imposées par
  multiplicateurs de Lagrange :
  - [Dirichlet](contraintes/dirichlet.md)
  - [Multi-points (MPC)](contraintes/mpc.md)
  - [Baignage (embedded)](contraintes/embedded.md)
  - [Contact (nœud-surface)](contraintes/contact.md)

Ce regroupement est la **nature physique** (`Physics`) que chaque variante
déclare : `Thermal` (conduction), `Radiation` (rayonnement, en plus de
`Thermal`), `Diffusion` (Fick), `Mechanical` (barre →
cadre 3D) et `Constraint` (les contraintes de Lagrange). On sélectionne les sous-modèles
d'une nature avec `model.filter(Physics::Mechanical)` (et les blocs d'une
matrice avec `k.filter(...)`) — voir
[Nature physique et filtrage](model.md#nature-physique-et-filtrage).

## Ce que chacune sait produire

La liste ci-dessus dit *où lire* chaque physique. Les tableaux qui suivent
disent **ce qu'elle sait produire** : les genres de matrice qu'elle déclare
([`MatrixKind`](ajouter-une-physique.md#un-genre-de-matrice--un-layout--un-noyau)),
la voie par laquelle elle obtient sa tangente, la nature de son intégration de
comportement, et la particularité de calcul qui la distingue.

Ils se lisent **en colonnes**. Une case remplie dit que la physique déclare ce
terme et que l'assembleur l'appelle ; un tiret dit qu'elle n'y contribue rien —
et **ce n'est pas un manque**. `matrix.mass(...)` sur un modèle contenant une
pression suiveuse voit simplement qu'elle n'a pas de masse.

Les **tags** donnés sous chaque nom sont les chaînes exactes que prennent les
constructeurs (`from_tag`) : c'est ce qu'on écrit, pas une paraphrase.

### Thermique — primale `T`, duale `q`, `filter("thermal")`

| Physique | `tangent` | `geometric` | `mass` | Comportement (`COMP`) | Particularité de calcul |
|---|---|---|---|---|---|
| [`heat_conduction`](thermique.md)<br>`isotropic` `orthotropic` `anisotropic` | — | — | capacité `ρ·cp` | flux `K·∇T` | Symétrie matériau **iso / ortho / aniso**. La conductivité isotrope est lue **par point de Gauss** — donc variable à l'intérieur d'une maille ; les constantes orientées le sont par maille. |
| [`convection`](thermique.md#convection-de-surface-robin--film) | — | — | — | `h·T` | **Aveugle à l'orientation** du maillage de bord : la normale est déjà consommée en écrivant `q·n = h(T − T_ext)`, et la mesure d'intégration est une magnitude. |
| [`radiation`](thermique.md#rayonnement-à-linfini-stefan-boltzmann) | **analytique**<br>`4σεT³` | — | — | `σε(T⁴ − T_∞⁴)`<br>**+** `ktan` | Non linéaire. La rigidité est le film **linéarisé autour de `T_∞`**, donc constante : c'est l'opérateur dont part Newton. La tangente porte la vraie non-linéarité. Seule physique à déclarer **deux natures**, `[Thermal, Radiation]`. |

### Diffusion — primale `c`, duale `j`, `filter("diffusion")`

| Physique | `tangent` | `geometric` | `mass` | Comportement (`COMP`) | Particularité de calcul |
|---|---|---|---|---|---|
| [`fick`](diffusion.md)<br>`isotropic` `orthotropic` `anisotropic` | — | — | stockage `poro` | flux `D·∇c` | Même opérateur que la conduction, **nature distincte** : partager un laplacien n'est pas partager une physique, et un problème couplé doit pouvoir sélectionner l'une sans l'autre. Flux nommé `j_*` pour cohabiter sans ambiguïté. |
| [`interface_transfer`](diffusion.md#transfert-à-travers-une-interface)<br>`kind=mass` `kind=thermal` | — | — | — | `h·(c₁ − c₂)` | **Quatre blocs** — deux diagonaux, deux `Coupling` dont les lignes vivent sur un maillage et les colonnes sur l'autre. Leur scatter est **séquentiel** : le coloriage qui rend le scatter parallèle sûr repose sur une seule connectivité. Conformité vérifiée jusqu'au nœud. |

### Mécanique — milieux continus, primales `u`, duales `f`, `filter("mechanical")`

| Physique | `tangent` | `geometric` | `mass` | Comportement (`COMP`) | Particularité de calcul |
|---|---|---|---|---|---|
| [`elasticity`](mecanique/elasticite.md)<br>`plane_stress` `plane_strain` `axisymmetric` `solid` · `isotropic` `orthotropic` `anisotropic` | **analytique**<br>c'est `K` | oui | oui | `σ = D·ε` | Loi linéaire, donc **la tangente *est* la rigidité**. Symétrie matériau iso/ortho/aniso : le repère d'orthotropie passe par des vecteurs du champ matériau, et la rotation du tenseur se fait à l'**ordre 4** plutôt que par une matrice de Bond. |
| [`plasticity`](mecanique/lois-plastiques.md) — 10 lois<br>`perfect` `isotropic` `drucker_prager` `ottosen` `gurson` `creep_norton` `creep_blackburn` `creep_lemaitre` `viscoplastic_chaboche` `viscoplastic_lemaitre_chaboche` | **analytique** ×2<br>**perturbation** ×8 | oui | oui | `σ`, `ε_p`, `p`<br>+ variables de la loi<br>+ `D_alg` (`ktan_i_j`) | La loi d'écoulement est un **attribut**. Tangente analytique pour les deux lois von Mises, **par perturbation** pour les huit autres — et toujours **symétrisée**. Les cinq lois visqueuses (Norton, Blackburn, Lemaitre, Chaboche…) **erronent sans `dt`** plutôt que d'intégrer comme si le temps n'existait pas. |
| [`damage`](mecanique/endommagement.md) — 3 lois<br>`mazars` `damage_tc` `sic_sic` | *aucune* | oui | oui | `σ`, `damage`<br>+ histoire de la loi | **Pas de tangente**, délibérément : l'opérateur d'itération reste la rigidité **non endommagée**. Damage TC porte deux histoires indépendantes — c'est ce qui laisse une fissure refermée reprendre toute sa charge, ce qu'un scalaire ne peut pas. |
| [`follower_pressure`](mecanique/pression-suiveuse.md) | — | — | — | traction `−p·n(u)` | **Aucune matrice** : ses `contributions` sont vides pour tous les genres. Toute son action passe par les forces internes, recalculées à chaque résidu. **Seule physique sensible au sens de parcours** du maillage de bord — l'orientation est la déclaration de ce qui est « dehors ». |

### Mécanique — structurel, efforts de section, `filter("mechanical")`

| Physique | `tangent` | `geometric` | `mass` | Comportement (`COMP`) | Particularité de calcul |
|---|---|---|---|---|---|
| [`truss`](mecanique/truss.md) | — | oui `N/L·P` | oui `ρA` | `N = E·A·ε` | Forme fermée **globale** : la direction vient des coordonnées, sans matrice de repère. Marche en 1-D, 2-D et 3-D sans changement. |
| [`bernoulli`](mecanique/bernoulli.md)<br>`aucun tag` | — | oui¹ | oui | `M`<br>`N, M`<br>`N, M_y, M_z, T` | Seule physique bâtie sur une interpolation **C¹** : elle exige un espace `HERMITE3`, dont la base cubique la rend **exacte aux nœuds** — un élément par barre suffit. La configuration (1-D, plan, spatial) se **déduit** de la dimension du maillage. Ne demande **ni `G` ni `A_s`** : réclamer une constante qu'une théorie n'utilise pas, c'est inviter la mauvaise. |
| [`timoshenko`](mecanique/timoshenko.md)<br>`aucun tag` | — | oui¹ | oui | `M, V`<br>`N, M, V`<br>`N, M_y, M_z, T, V_y, V_z` | **Une** physique pour les trois configurations, lues sur la dimension du maillage — elle remplace `frame` et `frame3d`. Élément **exact** (forme fermée en `Φ = 12EI/G·A_s·L²`), donc espace `MODEL_EMBEDDED` : la base dépend du matériau, aucun espace ne peut la tabuler. ¹ sauf en flexion pure, qui n'a pas d'effort axial. |
| [`shell`](mecanique/coques.md)<br>`thick` | — | — | — | `N_xx…N_xy`<br>`M_xx…M_xy`<br>`Q_xz, Q_yz` | **Multi-quadrature** — membrane et flexion au Gauss complet, cisaillement transverse en intégration réduite, ce qui empêche le blocage. **Six DDL** par nœud, les mêmes que la poutre en configuration spatiale, donc coque et portique partagent des nœuds sans adaptateur. Le vrillage est lié à la rotation de membrane, pas pénalisé : une pénalité diagonale s'opposerait à une rotation rigide, qui ne coûte rien. |

### Contraintes — multiplicateurs de Lagrange, `filter("constraint")`

| Physique | `tangent` | `geometric` | `mass` | Comportement (`COMP`) | Particularité de calcul |
|---|---|---|---|---|---|
| [`dirichlet`](contraintes/dirichlet.md) · [`mpc`](contraintes/mpc.md) · [`embedded`](contraintes/embedded.md) · [`contact`](contraintes/contact.md) | — | — | — | — | Aucun layout, rien d'intégré sur une maille : elles redéfinissent directement `contributions()` et rendent leurs blocs **C / Cᵀ** en `Literal`. L'assembleur reste sans le moindre cas particulier « Dirichlet ». |

### « Par perturbation » veut dire différences centrées

Pour les huit lois qui n'ont pas de tangente en forme fermée, `D_alg` est obtenu
en **perturbant la déformation** et en relançant le retour, composante par
composante :

\\[
D_{ij} \simeq \frac{\sigma_i(\varepsilon + h\\,e_j) - \sigma_i(\varepsilon - h\\,e_j)}{2h},
\qquad h = 10^{-6}\\,\lVert \varepsilon \rVert_\infty .
\\]

Six composantes de Voigt, deux évaluations chacune : **douze appels** au retour
par point de Gauss. Le pas doit rester bien au-dessus du bruit du retour (celui
d'Ottosen ou de Gurson converge à une tolérance, pas exactement) et bien en
dessous de l'échelle de courbure de la surface ; `1e-6·‖ε‖` tient
confortablement entre les deux. Les colonnes de cisaillement sont divisées par
deux en sortie, ce qui transforme `∂σ/∂ε_ij` en `∂σ/∂γ_ij` — la convention
ingénieur du reste du dépôt.

| voie | lois |
|---|---|
| **analytique** | `perfect`, `isotropic` (le module algorithmique J2) |
| **par perturbation** | `drucker_prager`, `ottosen`, `gurson`, les trois fluages, les deux Chaboche |

Le partage n'est pas une question de difficulté mais de **vérifiabilité** : seule
la forme fermée de von Mises a été confrontée à une différence finie et validée.
La dérivation analytique de Drucker-Prager, écrite d'abord, était fausse de 24 %
— plausible, et fausse ; seul l'oracle numérique l'a dit. Une tangente obtenue
par perturbation ne peut pas être mal dérivée, coûte douze évaluations d'une mise
à jour bon marché, et laisse Newton converger. C'est un bon échange.

Le rayonnement, lui, a bien une tangente **analytique** (`4σεT³ ∫NᵢNⱼ`) : sa
non-linéarité est une puissance scalaire d'une seule variable, pas une carte de
projection.

### Trois choses que ces tableaux ne disent pas

**Les forces internes ne suivent pas toujours `Bᵀσ`.** Le défaut est le noyau de
la mécanique des milieux continus, `f_i = ∫ ∂N_i/∂x · σ`. Une physique dont la
duale n'est pas un vecteur déplacement le redéfinit : la thermique et la
diffusion appliquent `Bᵀ` à un flux scalaire, tandis que la convection, le
rayonnement, le transfert d'interface et la **pression suiveuse** pondèrent par
`N` et non par `Bᵀ` — leur intégrande est une densité surfacique, pas une
grandeur conjuguée d'un gradient.

**La tangente stockée est symétrique.** `D_alg` transite par le champ d'état sous
forme de **triangle supérieur** (`ktan_i_j`, i ≤ j) et est relue en miroir : le
format ne peut structurellement pas porter une matrice non symétrique. Or un
écoulement **non associé** — Drucker-Prager, dont la dilatance diffère du
frottement — en a une. Elle est donc symétrisée, ce qui coûte à Newton son taux
quadratique sur cette loi, et rien d'autre. Voir
[Lois d'écoulement plastique](mecanique/lois-plastiques.md#la-tangente-cohérente-et-deux-limites-assumées).

**État absent ≠ état nul.** Au-delà de `ε_p` et `p`, une loi déclare l'état
qu'elle veut par `internal_names()` : la déformation primaire de Blackburn, la
contrainte de rappel de Chaboche (un tenseur complet), la porosité de Gurson, les
deux histoires de Damage TC. Le premier pas passe un vecteur **vide** — et non
un vecteur de zéros — pour qu'une loi démarrant d'une constante matériau puisse
faire la différence. Sans cela, un métal poreux démarrerait dense et ne
s'endommagerait jamais.

Pour **ajouter** une physique, voir [Ajouter une physique](ajouter-une-physique.md).
