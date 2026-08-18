# Conduction thermique

Cette page décrit la physique de **conduction thermique** (`HeatConduction`) et
la convection de surface associée. Elle suit le plan standard des physiques puis
la déroule sur un exemple complet — une ligne chauffée par une source à une
extrémité et maintenue à température fixe à l'autre — comparé à la **solution
analytique**.

Pour la mécanique générique du `Model` (orchestration, DOFs, assemblage), voir
[Modèle physique](model.md). Ici on se concentre sur le cas thermique.

## Équations continues résolues

En **régime stationnaire**, la forme forte est

\\[
-\nabla\cdot\big(k\\,\nabla T\big) = 0,
\\]

et en **régime transitoire**, l'équation de la chaleur porte un terme de
stockage :

\\[
\rho\\,c_p\\,\frac{\partial T}{\partial t} - \nabla\cdot(k\\,\nabla T) = Q.
\\]

La forme variationnelle de Galerkine (multiplication par une température
virtuelle, intégration par parties) fait apparaître la **rigidité** (conduction)
et, en transitoire, la **capacité** (stockage) — leurs formes discrètes
ci-dessous.

## Forme discrétisée

La conductivité donne, cellule par cellule, la matrice de **rigidité** :

\\[
K_{ij} = \int_K k(x)\\,\nabla N_i\cdot\nabla N_j\\,dx
\quad\approx\quad \sum_g k(\xi_g)\\,(\nabla N_i\cdot\nabla N_j)\big|_g\\,|J|_g\\,w_g
\\]

(implémentée dans `src/models/heat_conduction.rs`). En notant
\\( B = [\nabla N_1, \dots, \nabla N_n] \\) la matrice des gradients de forme
(taille \\( d\times n \\)), on a aussi \\( K = \int_\Omega k\\, B^\top B\\, d\Omega \\).
Le bloc local est écrit aux positions `row = (NodeId_i, "q")`,
`col = (NodeId_j, "T")`. Pour un SEG2 de longueur \\(L\\) et \\(k\\) uniforme on
retrouve la matrice analytique \\((k/L)\\,[[1,-1],[-1,1]]\\).

En transitoire, le terme de stockage discrétise en une **matrice de capacité**
(l'analogue thermique de la matrice de masse, Cast3M `CAPA`) :

\\[
C_{ij} = \int_\Omega \rho\\,c_p\\,N_i\\,N_j\\,d\Omega
\\;\approx\\; \sum_g \rho\\,c_p\\,N_i(\xi_g)\\,N_j(\xi_g)\\,|J|_g\\,w_g,
\\]

assemblée par [`assemble.mass`](operateurs/assemblage.md) (matériau `rho`, `cp`)
et concentrable en diagonale par [`lump`](operateurs/assemblage.md). Le système
semi-discret est \\( C\\,\dot T + K\\,T = F \\) ; l'intégration en temps
(θ-schéma, Euler implicite `(C/\Delta t + K)`) se pilote dans la couche Python.

## Variables et matériau

| | nom | rôle |
|---|---|---|
| **primale** (colonnes, inconnue) | `"T"` | température |
| **duale** (lignes, second membre) | `"q"` | flux de chaleur |
| **matériau** | `"k"` | conductivité (au point de Gauss) ; `rho`, `cp` **facultatifs** (capacité) |

La conductivité peut être **orientée** — voir
[Conduction orthotrope et anisotrope](#conduction-orthotrope-et-anisotrope) plus
bas ; `"k"` est alors remplacée par les constantes de la symétrie choisie.

## Mise en donnée (Rust, testé)

Le pipeline est toujours le même :

1. **`Coords`** — l'espace des nœuds (dimension géométrique).
2. **`Mesh`** — les éléments (ici des `SEG2` alignés sur \\([0,1]\\)).
3. **`FiniteElementSpace`** — l'interpolation (`lagrange1`).
4. **Matériau** — un `ElementField` portant la composante `"k"`, fabriqué
   commodément par `element_field::material_field(&model, &[("k", …)])` (les
   sous-modèles sans matériau, comme `Dirichlet`, sont ignorés).
5. **`Model`** — `Model::heat_conduction(&fes)`, composé par `|` (union) avec
   les conditions limites.
6. **Conditions limites :**
   - **Dirichlet** (`T` imposée) : un sous-modèle `Model::dirichlet` qui impose
     la valeur via multiplicateurs de Lagrange. L'utilisateur fournit le
     maillage des nœuds imposés et le maillage support des multiplicateurs —
     typiquement fabriqué depuis le premier avec le mesher générique
     [`barycenter`](operateurs/maillage.md) (nœuds neufs colocalisés). La valeur imposée
     \\(u_d\\) s'écrit dans le chargement au slot **`imposed_T`** du
     nœud-multiplicateur (cf. [Modèle physique](model.md)).
   - **Neumann / source** : une charge **ponctuelle** est une valeur du
     **chargement** sur la composante duale **`"q"`** au nœud concerné ; un
     **flux réparti** sur un bord (ou un volume) se transforme en charges
     nodales cohérentes par l'opérateur [`flux`](#exemple--un-carré) (analogue
     de `FLUX`/`PRES` de Cast3M).
7. **Assemblage + résolution** — `matrix::stiffness` puis le solveur
   `solver::lu::solve` (LU creuse directe, voir [Modèle physique](model.md)).

### Exemple : ligne chauffée

**Problème.** Sur \\([0,1]\\), une source de chaleur (flux de Neumann \\(Q\\)) est
appliquée en \\(x=0\\), et la température est imposée à \\(T=20\\) en \\(x=1\\).

**Solution analytique.** Sans génération volumique, \\(T''=0\\) : le profil est
linéaire. En notant \\(Q\\) le flux injecté et \\(k\\) la conductivité,

\\[
u(x) = 20 + \frac{Q}{k}\\,(1 - x).
\\]

De plus, le **multiplicateur de Lagrange** au nœud imposé (la *réaction* qui
maintient \\(T=20\\)) vaut exactement \\(Q\\) : tout le flux injecté en \\(x=0\\)
ressort en \\(x=1\\) — un bilan d'énergie discret.

**Code.** L'exemple ci-dessous est le test d'intégration
`tests/thermal_line.rs` : il est compilé **et exécuté à chaque `cargo test`**,
donc garanti à jour avec l'API.

```rust,ignore
{{#include ../../tests/thermal_line.rs:example}}
```

## Exemple Python

La **version Python** équivalente et documentée est dans le dépôt :
`examples/thermal_line_1d.py` (lancer avec `python examples/thermal_line_1d.py`
après `maturin develop`). Les compléments 2-D ci-dessous ont eux aussi leur
variante Python (`thermal_square_2d.py`, `thermal_convection_2d.py`).

## Compléments

### Exemple : un carré

La généralisation 2-D du cas précédent : un **carré** \\([0,1]^2\\) (grille
structurée de `QUA4`), chauffé par une source répartie sur le bord **gauche**
(\\(x=0\\)) et maintenu à \\(T=20\\) sur le bord **droit** (\\(x=1\\)). Les bords
haut et bas ne portent **aucune condition** : c'est la condition naturelle
(flux nul, bord *isolé*).

Comme les bords latéraux sont isolés, le champ ne dépend pas de \\(y\\) : le
carré **redonne le profil de la ligne**,

\\[
u(x) = 20 + \frac{Q}{k}\\,(1 - x),
\\]

et la **réaction totale** (somme des multiplicateurs sur le bord imposé) vaut le
flux injecté \\(Q\\).

**Mise en donnée d'un flux réparti.** Une source répartie se transforme en
**charges nodales cohérentes** \\(f_i = \int_\Gamma \varphi\\,N_i\\,d\Gamma\\) par
l'opérateur `flux` — l'analogue de `FLUX`/`PRES` de Cast3M. On lui donne le bord
(ici un maillage `SEG2`, intégré comme une **ligne** : la mesure vient du
Jacobien *manifold*) et la densité de flux (une constante, ou un champ par
éléments) ; il renvoie un `NodeField` sur la composante duale `"q"`, prêt à
composer (`|`) avec le reste du chargement. Sous le capot, pour un flux uniforme
sur des éléments linéaires, un nœud intérieur du bord reçoit \\(Q\\,h\\) et un
coin \\(Q\\,h/2\\) (somme \\(Q\\)) — mais on n'a plus à le calculer à la main.

```rust,ignore
{{#include ../../tests/thermal_square.rs:square}}
```

> Version Python : `examples/thermal_square_2d.py` (lancer avec
> `python examples/thermal_square_2d.py` après `maturin develop`).

### Conduction orthotrope et anisotrope

Un matériau feuilleté, fibré ou laminé ne conduit pas la chaleur de la même façon
dans toutes les directions. La conductivité devient alors un **tenseur** `K`, et
la rigidité

\\[
K_{ij} = \int_\Omega \nabla N_i^{\mathsf T}\\, \mathbf{K}\\, \nabla N_j \\, d\Omega
\\]

dont le cas isotrope `K = k·I` redonne le produit scalaire habituel.

C'est le **même axe de symétrie matériau** qu'en mécanique (chapitre
[Élasticité orthotrope](mecanique/orthotropie.md)), avec un tenseur d'ordre 2 au
lieu de 4 :

| symétrie | composantes matériau |
|---|---|
| `isotropic` (défaut) | `k` |
| `orthotropic` | `k_1`, `k_2`, `k_3` + le repère matériau |
| `anisotropic` | `k_11`, `k_12`, `k_13`, `k_22`, `k_23`, `k_33` + le repère |

Le repère est donné par des **vecteurs** — `V1X, V1Y` en 2-D, `V1X…V1Z,
V2X…V2Z` en 3-D — comme `MATE 'DIRECTION' V1 V2` de Cast3M. Ils sont
orthonormalisés en interne.

```python
model = pyrucast.Model.heat_conduction(fes, symmetry="orthotropic")
materials = pyrucast.element_field.material_field(
    model,
    [("k_1", 12.0), ("k_2", 3.0), ("k_3", 12.0), ("V1X", cos_a), ("V1Y", sin_a)],
)
```

La conductivité isotrope reste lue **au point de Gauss**, donc variable à
l'intérieur d'une maille ; les constantes orientées sont lues par maille, comme
les modules mécaniques.

L'exemple Rust est un **test de patch**, qui est ce qu'appelle une conductivité
orientée : un champ de température linéaire est harmonique pour *n'importe quel*
tenseur constant, donc l'imposer au bord doit le reproduire à l'intérieur quelle
que soit `K`. Le test ne s'arrête pas là — il relit le **flux** produit et le
compare à `K·∇T` calculé à la main, ce qui est le seul moyen de prendre la
rotation en défaut :

```rust,ignore
{{#include ../../tests/anisotropic_conduction.rs:example}}
```

Avec `∇T = (1, 0)`, le flux est la première colonne de `K` :
`K_xx = k₁cos²θ + k₂sin²θ` et `K_yx = (k₁ − k₂)·cosθ·sinθ`. Le terme
extra-diagonal n'est non nul que si le matériau est à la fois anisotrope **et**
désaligné — précisément le cas qu'une rotation fausse manquerait.

### Rayonnement à l'infini (Stefan-Boltzmann)

Une surface qui échange avec un environnement lointain à \\(T_\infty\\) rayonne

\\[
q\cdot n = \sigma\\,\varepsilon\\,\big(T^4 - T_\infty^4\big)
\\]

où \\(\sigma\\) est la constante de Stefan-Boltzmann et \\(\varepsilon\\)
l'émissivité. Primale `"T"`, duale `"q"` — les **mêmes** degrés de liberté que la
conduction, donc un bord rayonnant se couple directement dans sa rigidité, comme
la convection. Et comme elle, il n'a besoin d'**aucune normale** : la direction
est déjà consommée en écrivant `q·n`, il ne reste sous l'intégrale qu'un scalaire
et la mesure de surface.

#### Ce qui change par rapport à la convection : c'est non linéaire

La loi de Newton est linéaire en `T`, si bien que la convection ne contribue
qu'une matrice de film constante. `T⁴` ne l'est pas, d'où **trois** termes :

| terme | expression | rôle |
|---|---|---|
| rigidité | `4σεT_∞³ ∫ NᵢNⱼ dΓ` | le film radiatif **linéarisé**, un opérateur constant — le `h_r` classique |
| force interne | `∫ Nᵢ σε(T⁴ − T_∞⁴) dΓ` | le résidu, exact |
| tangente | `4σεT³ ∫ NᵢNⱼ dΓ` | la tangente cohérente à la température courante |

Linéariser la rigidité autour de \\(T_\infty\\) plutôt qu'autour de l'état
courant est ce qui la laisse être une matrice **constante** : c'est l'opérateur
dont on part pour une boucle de Newton, et à lui seul une itération de Picard
tout à fait utilisable. La **tangente** porte la vraie non-linéarité : elle relit
`T` dans l'état produit par l'intégration du comportement — le même couple
producteur/consommateur que le `D_alg` plastique.

#### Deux natures

Le rayonnement déclare `[Thermal, Radiation]`. Un bord rayonnant fait partie du
problème thermique — `filter("thermal")` doit le rendre — tandis que
`filter("radiation")` isole le terme non linéaire à part, pour l'assembler ou
l'inspecter seul. C'est le premier usage du caractère **ensembliste** de
`physics()`.

#### Unités

`sigma` vaut par défaut la constante SI, et `T` est alors une température
**absolue** (Kelvin) : une puissance quatrième n'a aucune invariance permettant
de translater une origine. Dans un autre système d'unités, fournir `sigma` comme
composante matériau.

```python
model = pyrucast.Model.heat_conduction(volume) | pyrucast.Model.radiation(bord)
materials = pyrucast.element_field.material_field(
    model, [("k", 20.0), ("emis", 0.8), ("T_inf", 300.0)]
)
```

#### Ce que ça vaut comme vérification

Deux choses se contrôlent sans acrobatie analytique : le flux rayonné doit valoir
exactement `σε(T⁴ − T_∞⁴)` fois l'aire, et la **tangente doit être la dérivée du
résidu**. Une loi en `T⁴` est précisément là où une tangente incohérente se cache
— Newton ramperait au lieu de converger quadratiquement — d'où sa comparaison à
une différence finie :

```rust,ignore
{{#include ../../tests/radiation.rs:example}}
```

### Convection de surface (Robin / film)

Le modèle `BoundaryTransfer` (`src/models/boundary_transfer.rs`) ajoute un **échange
convectif** avec un fluide à température ambiante \\(T_\text{ext}\\) sur un
bord : la loi de Newton du refroidissement

\\[
q\cdot n = h\\,\big(T - T_\text{ext}\big)
\\]

où \\(h\\) est le **coefficient d'échange** (film). Injectée dans le terme de
bord de la forme faible de la conduction, elle se scinde en deux ingrédients :

\\[
\underbrace{K_{ij} = h \int_\Gamma N_i\\,N_j\\,d\Gamma}_{\text{matrice de film (raideur)}}
\qquad
\underbrace{f_i = h\\,T_\text{ext} \int_\Gamma N_i\\,d\Gamma}_{\text{charge (second membre)}}
\\]

On le construit en lui passant les couples de variables à échanger — ici ceux de
la conduction, ce qui fait que le terme **se couple directement** dans la raideur
d'un `HeatConduction` :

```python
pyrucast.Model.boundary_transfer(bord_fes, [("T", "q")], "thermal")
```

| | nom | rôle |
|---|---|---|
| **primale** | `"T"` | température (partagée avec `HeatConduction`) |
| **duale** | `"q"` | flux de chaleur (partagé) |
| **matériau** | `"h_T"` | coefficient d'échange (film), nommé d'après la grandeur |

> Ce modèle n'a rien de thermique : la même loi décrit un transfert de masse en
> surface ou une fondation élastique, selon les composantes qu'on lui donne, et
> il partage son noyau avec le transfert d'interface. Voir
> **[Échanges](echanges.md)** pour la loi commune, la structure en quatre blocs
> et le choix entre un échange et une contrainte.

**Mise en donnée.** Le modèle fournit la matrice de film ; la part externe
\\(h\\,T_\text{ext}\\) est un **chargement**, bâti avec le **même** opérateur
[`flux`](#exemple--un-carré) que la source (densité \\(h\\,T_\text{ext}\\)). Le
terme de film rend la matrice **définie** : un problème purement Neumann +
convection est bien posé **sans Dirichlet**.

**Exemple.** Une dalle \\([0,1]^2\\) chauffée par un flux \\(Q\\) sur le bord
gauche et refroidie par convection sur le bord droit (haut/bas isolés). Tout le
flux ressort par convection, d'où le profil linéaire

\\[
T(x) = T_\text{ext} + \frac{Q}{h} + \frac{Q}{k}\\,(1 - x).
\\]

```rust,ignore
{{#include ../../tests/thermal_convection.rs:convection}}
```

> Version Python : `examples/thermal_convection_2d.py` (lancer avec
> `python examples/thermal_convection_2d.py` après `maturin develop`).
