# Calcul pas à pas (`pyrucast.stepping`)

`pyrucast.stepping` reproduit la procédure `PASAPAS` de Cast3M et celles
qu'elle appelle, une fonction Python par procédure : boucle en temps,
thermique (trois schémas, rayonnement, échange), mécanique non linéaire
(Newton modifié accéléré, sous-pas de convergence forcée, prédicteur), contact,
dynamique (Newmark), pilotages automatique et indirect, pas adaptatif, crochets
utilisateur. Le package est du Python pur, mais tout calcul sur un champ passe
par un opérateur pyrucast : seuls les trois scalaires de l'accélération
d'Anderson sont calculés en Python.

## Utilisation

On décrit le calcul par un dictionnaire, comme la table de `PASAPAS`, puis on
le lance :

```python
{{#include ../../tests/python/test_doc_pas_a_pas.py:donnees}}
```

```python
{{#include ../../tests/python/test_doc_pas_a_pas.py:lancement}}
```

Les résultats sont rangés dans `data["results"]`, un enregistrement par
instant de `save_times` (par défaut `times`), l'état initial compris :

```python
{{#include ../../tests/python/test_doc_pas_a_pas.py:resultats}}
```

Chaque enregistrement porte aussi `velocity` et `acceleration` (dynamique),
`load_multiplier` (pilotage indirect), `iterations`, `accelerations` et
`converged`. Il garde l'état calculé, pas ce qui s'en déduit : en grands
déplacements, `behaviour` contient la contrainte qu'écrit la loi (`S_*` ou
`T_*`), et la contrainte de Cauchy se calcule à partir de `displacement` et
`behaviour`, par `deformation(…, "gradient")` puis `stress_measure` (voir
[Grandes transformations](mecanique/grandes-transformations.md)).

### Les entrées

| clé | rôle | Cast3M |
|---|---|---|
| `model` | modèle complet : physiques, contraintes, charges portées | `MODELE` |
| `times` | instants calculés | `TEMPS_CALCULES` |
| `save_times` | instants enregistrés (défaut : `times`) | `TEMPS_SAUVES` |
| `materials` | champ matériau, ou `Evolution` de champs dans le temps | `CARACTERISTIQUES` |
| `material_laws` | lois scalaires de l'état, `E(T)`, `σ_y(T, φ)`… ([Évolution](evolution.md)) | `VARI` |
| `loads` | charges nodales et valeurs imposées, champ ou `Evolution` | `CHARGEMENT` |
| `temperature` | température imposée, sans modèle thermique | `CHARGEMENT 'T'` |
| `initial` | `displacement`, `behaviour`, `temperature`, `velocity`, `acceleration` | indice 0 des tables |
| `t_ref` | température sans dilatation | `TEMPERATURE_REFERENCE` |
| `hooks` | procédures utilisateur, par nom (`HOOKS`) | `REEV_MEC`, `PERSO1`, `AUTOPILO`… |

Les réglages et leurs valeurs par défaut sont ceux de `PAS_DEFA` :
`precision` et `strain_precision` (1e-4), `max_iterations` (49, 250 en
convergence forcée ou piloté), `delta_iterations` (10), `max_substeps` (100),
`forced_convergence` (vrai), `relaxation` (1), `acceleration` (vrai),
`tangent` et `tangent_every`, `predictor` (vrai), `adaptive_step`, `dynamics`,
`heat_scheme` (`nonlinear`, `linear`, `dupont`), `theta` (1),
`dupont_relaxation` (¼), `coherence` et `coherence_passes` (10), `automatic`,
`automatic_target` (1e-3), `indirect_control` (`{"load": F_p, "increment":
Δτ}`), `augmented_stiffness`, `geometric_stiffness` (vrai en grands
déplacements), `trace`. Une fonctionnalité absente du package (non local, MAN,
FEFP, frottement, rayonnement de cavité, changement de phase, modèle évolutif,
THM, advection) est **refusée** avec sa raison, jamais ignorée.

## L'algorithme

Les numéros d'algorithme suivent l'analyse de `PASAPAS`.

### Algo 1 — la boucle en temps (`step_by_step`)

```text
options ← resolve_options(data) ; problème ← describe_problem ; état ← initial_state
pour chaque instant cible TI de times
  pas adaptatif : instant ← adjust_time_step (Δt/√2 au-delà de 20 itérations,
                  Δt·√2 en deçà de 7, Δt/2 après un échec, sans dépasser TI)
  pour chaque passe (une ; jusqu'à coherence_passes avec coherence)
    thermique jusqu'à l'instant                                  // Algo 2
    mécanique jusqu'à l'instant, à la température atteinte       // Algo 3
    coherence : sortir si T n'a plus bougé d'une passe à l'autre
  échec : pas adaptatif → rejouer, Δt divisé par 2 ; sinon erreur
  t ← instant (pilotage automatique : t ← t + α (instant − t), et rejouer TI)
  store_step ; crochet after_step
```

### Algo 2 — la thermique (`heat_step`, `heat.heat_step_linear`, `heat.heat_step_dupont`)

L'inconnue porte la température et les multiplicateurs des températures
imposées. Le schéma θ itéré (`TRANSNON`) résout

\\[
\Bigl(\theta K(T_n) + \tfrac{C}{\Delta t}\Bigr)\,\delta T
= F - F_{int}(T_\theta) - \tfrac{C}{\Delta t}(T - T_n),
\qquad T_\theta = T_n + \theta (T - T_n),
\\]

l'opérateur étant figé à `T_n` (la tangente : conduction, et `4σεT³` pour le
rayonnement), le résidu exact, les matériaux réévalués à `T_θ`. Arrêt quand la
correction est sous `precision` relativement à la température ; accélération
d'Anderson non pondérée aux itérations paires à partir de la quatrième. Sans
`cp`, le problème est stationnaire (`θ = 1`). `linear` fait une seule
résolution, matériaux du début ; `dupont` résout le schéma à trois niveaux

\\[
\Bigl(\tfrac{C}{\Delta t} + (\tfrac12+\lambda)K\Bigr) T_{n+1}
= \Bigl(\tfrac{C}{\Delta t} + (2\lambda-\tfrac12)K\Bigr) T_n - \lambda K T_{n-1} + F ,
\\]

linéarisé à la température extrapolée au milieu du pas.

### Algo 3 — le pas mécanique (`mechanical_step`)

L'inconnue `x` porte le déplacement **et** les multiplicateurs des
contraintes. Chaque résolution rend une correction de déplacement et les
multiplicateurs **en total** — les réactions courantes, dont le solveur
unilatéral teste le signe :

\\[
\begin{bmatrix} K & C^\mathsf{T} \\ C & 0 \end{bmatrix}
\begin{bmatrix} \delta u \\ \lambda \end{bmatrix}
=
\begin{bmatrix} F_{ext} - F_{int}(u) \\ g - C u \end{bmatrix},
\qquad r = F_{ext} - F_{int}(u) - C^\mathsf{T}\lambda .
\\]

```text
pour chaque sous-pas (max_substeps au plus)
  données du pas à l'instant visé : matériaux, charges, ε_th = α (T − T_ref)
  K ← iteration_operator : rigidité élastique gardée tant que le matériau ne
      change pas (tangente cohérente avec `tangent`, + 4/Δt² M en dynamique)
  R⁰ au début du pas ; δx⁰ = K⁻¹R⁰ ; F_ref ← reference_forces (1er sous-pas)
  prédicteur : R⁰ += a·F_NL du pas précédent (nonlinear_predictor)
  equilibrium_iterations                                         // Algo 4
  convergé : fin ; sinon (convergence forcée) l'itéré atteint devient le
  départ d'un nouveau sous-pas, vers une cible rapprochée si relaxation < 1
```

`F_ref = |δx⁰·(F_ext ; −g)| / ‖δu⁰‖∞ + 2 F_s`, borné par la plus grande force
appliquée (`F_s` : la plus grande force interne au début du pas).
`F_NL = K Δx − ΔR⁰ + r` mesure ce que l'opérateur n'a pas prévu ; le pas
suivant en reprend la part `a`, rapport des incréments de charge projetés,
borné par `Δt/Δt_prec`.

### Algo 4 — les itérations d'équilibre (`mechanics.equilibrium_iterations`)

```text
pour i = 1, 2, …
  tangent : K ← tangente à l'itéré, toutes les tangent_every itérations
  i > 3 et pas de correction à l'itération précédente :
      R ← R − correction d'Anderson                              // ACT3
  (δu, λ) ← K⁻¹R (unilatéral) ; u ← u + δu
  pilotage : rééchelonner δu et le niveau de charge
  lois intégrées à u                                             // Algo 5
  R, r, réactions ← bilan ; tests de convergence_status          // Algo 6
```

L'accélération (`anderson.anderson_correction`, mémoire 3) minimise
`‖F₀ + Σ xⱼ (Fⱼ − F₀)‖` sur les quatre derniers résidus pondérés par
`(Σ mesures des mailles autour du nœud)^-½` ; elle recule à deux directions,
une, aucune, comme `ACT3`, et une correction qui reviendrait en arrière le long
du dernier incrément est abandonnée.

### Algo 5 — les lois (`mechanics.integrate_increment`)

La déformation que lit chaque loi vient de l'opérateur `strain` du modèle,
moins la dilatation thermique ; les lois avancent depuis l'état convergé du
début du sous-pas (`integrate_behavior`). Toutes calculent leur contrainte à
partir de leur déformation élastique avec le matériau de fin de pas
(voir [comportement](operateurs/comportement.md)) : un module qui dépend de la
température ne demande donc **aucune** correction, là où Cast3M ajoute
`(D⁻¹ₙ₊₁ − D⁻¹ₙ)σₙ` à l'incrément.

### Algo 6 — les critères (`mechanics.convergence_status`)

Convergé quand le résidu des forces — et la variation des réactions — sous
`F_ref` est sous `precision`, la variation de déformation de l'itération sous
`strain_precision`, et le résidu des moments sous `M_ref` sous la précision
des moments. Non-convergence détectée quand le résidu n'a pas baissé de 1 % en
`delta_iterations` itérations, ou au plafond d'itérations : le plafond triple,
et l'échec est déclaré quand la déformation ne bouge plus. Rien n'est détecté
sous pilotage.

### Dynamique, contact, pilotages

- **Newmark** (γ = ½, β = ¼) : `(K + 4/Δt² M) δu = F − F_int − M a(u)`, avec
  `a = 4/Δt² (u − uₙ) − 4/Δt vₙ − aₙ` ; il conserve l'énergie d'une vibration
  libre linéaire.
- **Contact** : une contrainte d'inégalité (`sense=">="` ou `Model.contact`) est
  résolue par le solveur à statuts à chaque itération ; les multiplicateurs
  en total en portent le signe.
- **Automatique** (`AUTOPILO`) : `Δu` est mis à l'échelle pour que sa
  déformation vaille `automatic_target`, et la fraction de charge
  `α ← α − (r·Δu)/(R·Δu)` ; le temps n'avance que de `α`. Le crochet
  `automatic_measure(data, state, Δu, αΔu_él)` remplace cette taille, comme la
  redéfinition d'`AUTOPILO` : une taille **négative** retourne `Δu`. Une
  grandeur choisie pour croître — la flèche d'un sommet — fait ainsi franchir
  les points limites, où la réponse de l'opérateur change de sens ; la
  déformation, toujours positive, ne le peut pas (Cast3M non plus, dont le cas
  `snap.dgibi` redéfinit `AUTOPILO` de la même façon).
- **Indirect** (`PILOINDI`) : `F = F_ext + η F_p`, `η` choisi pour que la
  déformation avance de `Δτ` dans sa propre direction au point le plus
  sollicité.

### Grands déplacements

Il n'y a pas de clé à activer : les lois mécaniques construites avec
`strain_measure="green_lagrange"` ou `"hencky"` déclarent leur mesure, et
`describe_problem` la lit. Le calcul suit alors la voie du lagrangien total
(cf. [Grandes transformations](mecanique/grandes-transformations.md)) :

```text
describe_problem : mesure lue sur les lois ; jeu « current » du Coords (nommé
                   « stepping:current »), ajouté au premier calcul, retrouvé ensuite
pour chaque itéré u :
  E ← strain(u) ; S ← loi(E)                     sur la référence
  P ← stress_measure(S, F(u), "pk1")
  r ← F_ext − internal_forces(P) − Cᵀλ           sur la référence
      + pression suiveuse évaluée en x = X + u   sur « current »
opérateur : stiffness + geometric(σ)              sur « current », en x = X + u
  + 4/Δt² M en dynamique                         sur la référence (ρ₀ dV₀)
  construit en début de sous-pas, puis à la 2ᵉ itération,
  puis toutes les max(delta_iterations, 20) itérations ; au-delà de 20, échec
thermique : conduction, échanges, rayonnement et flux sur « current »,
            en x = X + uₙ (début de pas), axes matériau tournés par R ;
            capacité sur la référence (ρ₀ c dV₀) ; l'opérateur additionne
            les matrices des deux
résultats : "behaviour" garde S (ou T) ; σ, à la demande, par stress_measure
```

L'opérateur est celui de Cast3M, `K_e(GEOM2) + K_σ` : il ne règle que la
vitesse de convergence, la solution étant fixée par le résidu. Il est
reconstruit dès la deuxième itération, la première correction ayant déplacé la
géométrie (UNPAS reconstruit aussi à `i = 2`). `geometric_stiffness=False` le
retire ; demandé en petits déplacements, il est refusé.

La conductivité donnée est celle de la matière telle qu'elle est : la
thermique se calcule sur la pièce déformée, les axes d'une conduction orientée
tournant avec elle (`rotate_frame`, rotation `R` de `F = R U`), et le flux de
la loi est le flux réel. Seule la capacité reste sur la référence, où elle est
exacte (cf. [Grandes transformations](mecanique/grandes-transformations.md#la-conduction)).

Sont refusés en grands déplacements : le contact (ses relations devraient
suivre le glissement du point projeté), une loi de petites déformations à côté
d'une loi en grandes transformations (poutres et coques n'ont pas encore de
formulation en grandes transformations ; une barre lit `green_lagrange`, cf.
[Les barres et les câbles](mecanique/grandes-transformations.md#les-barres-et-les-câbles)).

## Écrire sa propre boucle

Chaque procédure s'appelle seule ; `step_by_step` n'est que leur
enchaînement :

```python
{{#include ../../tests/python/test_doc_pas_a_pas.py:procedures}}
```

## Correspondance avec Cast3M

| Cast3M | pyrucast |
|---|---|
| `PASAPAS` | `stepping.step_by_step` |
| `PAS_DEFA` | `stepping.resolve_options` |
| `PAS_MODL` | `stepping.describe_problem` |
| `PAS_INIT` | `stepping.initial_state` |
| `PAS_ETAT`, `PAS_MATE` | `stepping.materials_at` |
| `PAS_RESU` | `stepping.store_step` |
| `PAS_AJUSTE` | `stepping.adjust_time_step` |
| `TRANSNON`, `TRANSLIN`, `DUPONT2` | `stepping.heat_step`, `stepping.heat.heat_step_linear`, `stepping.heat.heat_step_dupont` |
| `UNPAS` | `stepping.mechanical_step` |
| opérateur initial, `F_ref`, `F_NL` | `stepping.mechanics.iteration_operator`, `reference_forces`, `nonlinear_predictor` |
| itérations, `COMP`, critères | `stepping.mechanics.equilibrium_iterations`, `integrate_increment`, `convergence_status` |
| `ACT3` | `stepping.anderson.anderson_correction` |
| `AUTOPILO`, `PILOINDI` | `stepping.control.automatic_control` (ou le crochet `automatic_measure`), `indirect_control` |
| `REEV_MEC`, `REEV_THE`, `PERSO1`, `CHARMECA`, `CHARTHER`, `PARATHER` | crochets `after_mechanics`, `after_heat`, `after_step`, `extra_mechanical_load`, `extra_heat_load`, `heat_material_update` |
| `GRANDS_DEPLACEMENTS`, `LAGRANGIEN` | la mesure des lois (`strain_measure`) ; toujours le lagrangien total |
| `K_SIGMA` | `geometric_stiffness` |
| `FORM` (GEOM2) | `stepping.configuration.Configurations` |

## Écarts assumés

- **Lois sécantes, sans correction de module** (ci-dessus).
- **Dupont à trois niveaux d'un pas à l'autre** : `T_{n−1}` est la vraie
  température précédente ; `DUPONT2` dégénère en θ = ¾ entre deux pas.
- **Rayonnement** : l'opérateur est la tangente `4σεT³` à `T_n` plutôt que la
  sécante `σε(T² + T∞²)(T + T∞)` ; le résidu est exact dans les deux cas, et le
  schéma `linear` linéarise le rayonnement au lieu de l'ignorer.
- **Un pas adaptatif raté n'est pas gardé** : il est rejoué, réduit, depuis le
  dernier état convergé.
- **La température initiale** vaut `t_ref` quand elle n'est pas donnée.
- **La tangente** (`tangent`) est la tangente cohérente de la loi, là où
  `KTAN` donne la tangente continue.
- **La rigidité augmentée** est une matrice donnée (`augmented_stiffness`) ;
  l'augmentation automatique n'est pas offerte.
- **Grands déplacements en lagrangien total**, pour toute loi : ni Δε de
  mi-pas (GEOMM) ni transport des contraintes, l'état restant sur la
  référence. Le prédicteur HPP n'est pas repris.
- **Pas de prédicteur `F_NL` en grands déplacements** : une rotation finie
  n'est pas dans le noyau de l'opérateur, si bien que `K Δx` y lit des efforts
  là où rien ne se déforme, et l'opérateur change d'un pas à l'autre.
- **Les axes d'une conduction orientée en grands déplacements** tournent de
  la rotation `R` de `F = R U`. Cast3M les range dans le repère local de
  chaque élément, recalculé sur la géométrie courante : les deux s'accordent
  sous une rotation rigide, pas sous un cisaillement. La capacité est calculée
  sur la référence, `ρ₀ cp dV₀`, où elle est exacte.
