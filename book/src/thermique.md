# Conduction thermique

Cette page décrit la physique de **conduction thermique** (`HeatConduction`),
sa **mise en donnée**, et la déroule sur un exemple complet — une ligne
chauffée par une source à une extrémité et maintenue à température fixe à
l'autre — dont le résultat est comparé à la **solution analytique**.

Pour la mécanique générique du `Model` (orchestration, DOFs, assemblage), voir
[Modèle physique](model.md). Ici on se concentre sur le cas thermique.

## Le modèle `HeatConduction`

Régime stationnaire, forme forte :

\\[
-\nabla\cdot\big(k\,\nabla T\big) = 0
\\]

La forme variationnelle de Galerkine donne, cellule par cellule, la matrice de
rigidité :

\\[
K_{ij}^{(\text{loc})} = \int_K k(x)\,\nabla N_i\cdot\nabla N_j\,dx
\quad\approx\quad \sum_g k(\xi_g)\,(\nabla N_i\cdot\nabla N_j)\big|_g\,|J|_g\,w_g
\\]

(implémenté dans `src/models/heat_conduction.rs`). Les conventions de nommage :

| | nom | rôle |
|---|---|---|
| **primale** (colonnes, inconnue) | `"T"` | température |
| **duale** (lignes, second membre) | `"q"` | flux de chaleur |
| **matériau** | `"k"` | conductivité (au point de Gauss) |

Le bloc local est écrit aux positions `row = (NodeId_i, "q")`,
`col = (NodeId_j, "T")`. Pour un SEG2 de longueur \\(L\\) et \\(k\\) uniforme on
retrouve la matrice analytique \\((k/L)\,[[1,-1],[-1,1]]\\).

## Mise en donnée

Le pipeline est toujours le même :

1. **`Coords`** — l'espace des nœuds (dimension géométrique).
2. **`Mesh`** — les éléments (ici des `SEG2` alignés sur \\([0,1]\\)).
3. **`FiniteElementSpace`** — l'interpolation (`lagrange1`).
4. **Matériau** — un `ElementField` portant la composante `"k"`, fabriqué
   commodément par `build::material_field(&model, &[("k", …)])` (les
   sous-modèles sans matériau, comme `Dirichlet`, sont ignorés).
5. **`Model`** — `Model::heat_conduction(&fes)`, composé par `+` avec les
   conditions limites.
6. **Conditions limites :**
   - **Dirichlet** (`T` imposée) : un sous-modèle `Model::dirichlet` qui impose
     la valeur via multiplicateurs de Lagrange. L'utilisateur fournit le
     maillage des nœuds imposés et le maillage support des multiplicateurs —
     typiquement fabriqué depuis le premier avec le mesher générique
     [`barycenter`](mesh.md) (nœuds neufs colocalisés). La valeur imposée
     \\(u_d\\) s'écrit dans le chargement au slot **`imposed_T`** du
     nœud-multiplicateur (cf. [Modèle physique](model.md)).
   - **Neumann / source** : une charge **ponctuelle** est une valeur du
     **chargement** sur la composante duale **`"q"`** au nœud concerné ; un
     **flux réparti** sur un bord (ou un volume) se transforme en charges
     nodales cohérentes par l'opérateur [`flux`](#exemple--un-carré) (analogue
     de `FLUX`/`PRES` de Cast3M).
7. **Assemblage + résolution** — `assemble::stiffness` puis le solveur dense
   `solve` (voir [Modèle physique](model.md)).

## Exemple : ligne chauffée

**Problème.** Sur \\([0,1]\\), une source de chaleur (flux de Neumann \\(Q\\)) est
appliquée en \\(x=0\\), et la température est imposée à \\(T=20\\) en \\(x=1\\).

**Solution analytique.** Sans génération volumique, \\(T''=0\\) : le profil est
linéaire. En notant \\(Q\\) le flux injecté et \\(k\\) la conductivité,

\\[
u(x) = 20 + \frac{Q}{k}\,(1 - x).
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

> La **version Python** équivalente et documentée est dans le dépôt :
> `examples/thermal_line_1d.py` (lancer avec
> `python examples/thermal_line_1d.py` après `maturin develop`).

## Exemple : un carré

La généralisation 2-D du cas précédent : un **carré** \\([0,1]^2\\) (grille
structurée de `QUA4`), chauffé par une source répartie sur le bord **gauche**
(\\(x=0\\)) et maintenu à \\(T=20\\) sur le bord **droit** (\\(x=1\\)). Les bords
haut et bas ne portent **aucune condition** : c'est la condition naturelle
(flux nul, bord *isolé*).

Comme les bords latéraux sont isolés, le champ ne dépend pas de \\(y\\) : le
carré **redonne le profil de la ligne**,

\\[
u(x) = 20 + \frac{Q}{k}\,(1 - x),
\\]

et la **réaction totale** (somme des multiplicateurs sur le bord imposé) vaut le
flux injecté \\(Q\\).

**Mise en donnée d'un flux réparti.** Une source répartie se transforme en
**charges nodales cohérentes** \\(f_i = \int_\Gamma \varphi\,N_i\,d\Gamma\\) par
l'opérateur `flux` — l'analogue de `FLUX`/`PRES` de Cast3M. On lui donne le bord
(ici un maillage `SEG2`, intégré comme une **ligne** : la mesure vient du
Jacobien *manifold*) et la densité de flux (une constante, ou un champ par
éléments) ; il renvoie un `NodeField` sur la composante duale `"q"`, prêt à
composer (`+`) avec le reste du chargement. Sous le capot, pour un flux uniforme
sur des éléments linéaires, un nœud intérieur du bord reçoit \\(Q\,h\\) et un
coin \\(Q\,h/2\\) (somme \\(Q\\)) — mais on n'a plus à le calculer à la main.

```rust,ignore
{{#include ../../tests/thermal_square.rs:square}}
```

> Version Python : `examples/thermal_square_2d.py` (lancer avec
> `python examples/thermal_square_2d.py` après `maturin develop`).
