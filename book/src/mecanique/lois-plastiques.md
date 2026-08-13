# Lois d'écoulement plastique

## Introduction

La [plasticité parfaite](plasticite.md) est un cas particulier. Toutes les lois
élastoplastiques de pyrucast partagent **la même physique** — mêmes degrés de
liberté, même rigidité élastique comme opérateur d'itération, même montage
incrémental A → B, même état interne — et ne diffèrent que par leur **surface de
charge** et leur règle d'écoulement.

La loi est donc un **attribut** du modèle de plasticité, pas un modèle à part.
C'est la convention de Cast3M, où `PLASTIQUE PARFAIT`, `PLASTIQUE ISOTROPE`,
`PLASTIQUE DRUCKER_PRAGER` et `PLASTIQUE OTTOSEN` sont des variantes d'une seule
formulation.

| loi | surface | ce qu'elle capture | matériau |
|---|---|---|---|
| `plasticity_perfect` | `q = σ_y` | métal sans écrouissage | `E, nu, sigma_y` |
| `plasticity_isotropic` | `q = σ_y + H·p` | métal écrouissable | `+ H` |
| `drucker_prager` | `q + α·I₁ = k` | sols, roches, poudres — sensibilité à la pression | `E, nu, alpha, k, psi` |
| `ottosen` | 4 paramètres, dépendante de l'angle de Lode | béton — traction ≠ compression | `E, nu, a, b, k_1, k_2, sigma_c` |

Ce qui est mutualisé dans `src/models/plastic/` : le prédicteur élastique, la
**boucle sécante des contraintes planes** (qui résout `σ_zz(B) = 0` *autour* de
n'importe quelle loi, si bien qu'aucune loi ne connaît les contraintes planes),
le retour par **plan sécant** pour les surfaces sans forme fermée, et la tangente
cohérente.

L'état est toujours porté en **3-D complet** (six `eps_p_*` et un `p` cumulé)
quel que soit le modèle 2-D : chaque retour est ainsi identique en contraintes
planes, déformations planes, axisymétrie et massif — seules les projections
d'entrée et de sortie changent.

## Équations continues résolues

Les quatre lois partagent le cadre de l'élastoplasticité en **petites
déformations**, et n'en particularisent que deux fonctions — la surface `f` et
le potentiel `g` :

- **partition** : \\( \varepsilon = \varepsilon^e + \varepsilon^p \\) ;
- **élasticité** : \\( \sigma = D : \varepsilon^e = D : (\varepsilon - \varepsilon^p) \\) ;
- **domaine élastique** : \\( f(\sigma, \mathcal V) \le 0 \\), où \\( \mathcal V \\)
  rassemble les variables d'écrouissage ;
- **règle d'écoulement** : \\( \dot\varepsilon^p = \dot\lambda\\,\dfrac{\partial g}{\partial\sigma} \\),
  l'écoulement étant dit **associé** si \\( g = f \\) ;
- **écrouissage** : \\( \dot{\mathcal V} = \dot\lambda\\,h(\sigma, \mathcal V) \\) ;
- **Kuhn-Tucker** : \\( \dot\lambda \ge 0 \\), \\( f \le 0 \\), \\( \dot\lambda\\,f = 0 \\),
  et la **consistance** \\( \dot\lambda\\,\dot f = 0 \\) tant que l'on plastifie.

Toutes les surfaces s'écrivent sur les invariants de la contrainte :

\\[
I_1 = \operatorname{tr}\sigma = 3\sigma_m, \qquad
s = \sigma - \sigma_m\\,I, \qquad
J_2 = \tfrac12\\,s\\!:\\!s, \qquad
J_3 = \det s,
\\]

d'où la contrainte équivalente \\( q = \sqrt{3J_2} \\) et l'**angle de Lode**
\\( \theta \\), qui repère la direction *dans* le plan déviatorique :

\\[
\cos 3\theta = \frac{3\sqrt3}{2}\\,\frac{J_3}{J_2^{3/2}}, \qquad \theta \in [0, \pi/3].
\\]

C'est le jeu d'invariants retenu qui classe les quatre lois : von Mises ne voit
que \\( q \\) ; Drucker-Prager ajoute \\( I_1 \\), donc la pression ;
Ottosen ajoute en plus \\( \theta \\), donc la direction déviatorique.

## Forme discrétisée — le problème incrémental

Sur un pas A → B, les conditions de Kuhn-Tucker deviennent un problème de
**projection** : trouver \\( \Delta\lambda \ge 0 \\) tel que

\\[
\sigma_B = D : \Big(\varepsilon_B - \varepsilon^p_A - \Delta\lambda\\,
\frac{\partial g}{\partial\sigma}\Big),
\qquad f(\sigma_B, \mathcal V_B) = 0 .
\\]

Le **prédicteur élastique** gèle la plasticité sur le pas :

\\[
\sigma^{\text{tr}} = D : (\varepsilon_B - \varepsilon^p_A).
\\]

Si \\( f(\sigma^{\text{tr}}, \mathcal V_A) \le 0 \\), le pas est élastique et
l'état interne ne bouge pas. Sinon il faut **retourner** sur la surface — et
c'est là, et seulement là, que les lois diffèrent.

## Écrouissage isotrope

La surface se dilate avec la déformation plastique cumulée :

\\[
f(\sigma, p) = q - \big(\sigma_y + H\\,p\big), \qquad
\frac{\partial f}{\partial\sigma} = \frac{3}{2}\\,\frac{s}{q}, \qquad
\dot p = \dot\lambda .
\\]

La normale est **colinéaire au déviateur** et de trace nulle : le retour est
*radial* et la plasticité isochore. Le déviateur d'essai n'étant que remis à
l'échelle, \\( q_B = q^{\text{tr}} - 3\mu\\,\Delta p \\), la consistance devient
une équation **affine** dont la solution est fermée :

\\[
q^{\text{tr}} - 3\mu\\,\Delta p = \sigma_y + H\\,(p_A + \Delta p)
\quad\Longrightarrow\quad
\Delta p = \frac{q^{\text{tr}} - \sigma_y(p_A)}{3\mu + H},
\\]

la mise à jour étant alors

\\[
s_B = s^{\text{tr}}\Big(1 - \frac{3\mu\\,\Delta p}{q^{\text{tr}}}\Big), \qquad
\varepsilon^p_B = \varepsilon^p_A + \frac{3\\,\Delta p}{2\\,q^{\text{tr}}}\\,s^{\text{tr}},
\qquad \sigma_m \ \text{inchangé.}
\\]

`H = 0` redonne exactement la loi parfaite — un seul chemin de code sert les
deux, ce que vérifie un test.

**Sa tangente cohérente est la seule analytique** du lot, le module
algorithmique classique évalué au prédicteur :

\\[
D_{\text{alg}} = K\\,I \otimes I + 2\mu\\,\theta\\,\mathbb I_{\text{dev}}
- 2\mu\\,\bar\theta\\;\hat n \otimes \hat n,
\\]
\\[
\theta = \frac{\sigma_y(p_A + \Delta p)}{q^{\text{tr}}}, \qquad
\bar\theta = \frac{3\mu}{3\mu + H} - (1 - \theta), \qquad
\hat n = \frac{s^{\text{tr}}}{\lVert s^{\text{tr}}\rVert}.
\\]

Le facteur \\( \theta \\) est ce qui distingue le module **algorithmique** du
module élastoplastique *continu* : il rend compte du pas **fini**, et l'omettre
coûterait à Newton sa convergence quadratique. À \\( H = 0 \\) les deux
coefficients coïncident (\\( \bar\theta = \theta \\)) et l'on retrouve la
tangente parfaite : l'écrouissage coûte un terme, pas une seconde dérivation.

## Drucker-Prager

Sols, roches, bétons et poudres sont **plus résistants en compression qu'en
traction** : leur seuil dépend de la pression hydrostatique, que von Mises
ignore. Drucker-Prager est le cône le plus simple qui le capture :

\\[
f(\sigma) = q + \alpha\\,I_1 - k
\\]

### Un écoulement non associé

Un écoulement associé sur ce cône ferait dilater le matériau sous cisaillement
d'exactement ce que son frottement implique — bien trop pour un milieu granulaire
réel. Le potentiel plastique porte donc **sa propre** pente, la dilatance `ψ` :

\\[
g(\sigma) = q + \psi\\,I_1, \qquad \psi \le \alpha
\\]

`ψ = α` redonne l'écoulement associé ; `ψ = 0` donne un écoulement plastique
isochore à résistance frottante.

### Le retour sur le flanc reste fermé

La normale au potentiel se sépare en une part déviatorique et une part
sphérique :

\\[
\frac{\partial g}{\partial\sigma} = \frac{3}{2}\\,\frac{s}{q} + \psi\\,I .
\\]

L'opérateur élastique envoie la première sur \\( 3\mu \\) dans `q` et la seconde
sur \\( 9K\psi \\) dans `I₁`, si bien que les deux invariants sont **affines** en
le multiplicateur :

\\[
q_B = q^{\text{tr}} - 3\mu\\,\Delta\lambda, \qquad
I_{1,B} = I_1^{\text{tr}} - 9K\psi\\,\Delta\lambda,
\\]

et la consistance \\( f(\sigma_B) = 0 \\) se résout, comme en J2, sans itérer :

\\[
\Delta\lambda = \frac{q^{\text{tr}} + \alpha I_1^{\text{tr}} - k}{3\mu + 9K\alpha\psi},
\qquad
\Delta\varepsilon^p = \frac{3\\,\Delta\lambda}{2\\,q^{\text{tr}}}\\,s^{\text{tr}}
+ \psi\\,\Delta\lambda\\,I .
\\]

Le terme \\( 9K\alpha\psi \\) au dénominateur est **le** couplage
pression-cisaillement : c'est là que la dilatance rigidifie (ou non) la réponse,
et il disparaît dès que \\( \psi = 0 \\).

### Le sommet

Un cône a une pointe, en \\( I_1 = k/\alpha \\), \\( s = 0 \\). Le critère qui la
détecte est celui-là même que donne la formule fermée : si elle rend
\\( q_B < 0 \\), le retour lisse a *dépassé* l'axe hydrostatique et la solution
n'est pas admissible. La contrainte s'effondre alors sur la pointe,

\\[
s_B = 0, \qquad \sigma_{m,B} = \frac{k}{3\alpha},
\\]

tout ce que le prédicteur avait construit au-delà devenant plastique :

\\[
\Delta\varepsilon^p = \frac{s^{\text{tr}}}{2\mu}
+ \frac{I_1^{\text{tr}} - k/\alpha}{9K}\\,I,
\qquad \Delta p = \frac{q^{\text{tr}}}{3\mu}.
\\]

C'est **le** cas qu'une implémentation naïve rate silencieusement sous forte
traction. Avec \\( \alpha = 0 \\) le cône est un cylindre, il n'a pas de sommet,
et cette branche est inatteignable — le retour sur le flanc réussit toujours,
puisque c'est von Mises.

Au sommet la contrainte est figée, donc **la tangente y est nulle** — et
délibérément. Renvoyer le module élastique, la solution de facilité, rendrait la
tangente incohérente avec le retour. Un corps entièrement à son sommet assemble
donc une tangente singulière : ce n'est pas un artefact, c'est le constat honnête
qu'un tel matériau ne porte plus rien.

## Ottosen

Le béton casse très différemment en traction et en compression, et sa résistance
sous une pression donnée dépend de **la direction déviatorique** de la
contrainte. Ni von Mises (aveugle à la pression) ni Drucker-Prager (aveugle à
cette direction) ne le capturent. La surface d'Ottosen le fait par une dépendance
à l'**angle de Lode** :

\\[
f(\sigma) = a\frac{J_2}{\sigma_c^2} + \lambda(\theta)\frac{\sqrt{J_2}}{\sigma_c}
          + b\frac{I_1}{\sigma_c} - 1
\\]

où la fonction de forme déviatorique vaut, selon le signe de \\( \cos 3\theta \\),

\\[
\lambda(\theta) =
\begin{cases}
k_1\\,\cos\\!\Big[\tfrac13\arccos\big(k_2\cos 3\theta\big)\Big]
& \text{si } \cos 3\theta \ge 0,\\\\
k_1\\,\cos\\!\Big[\tfrac{\pi}{3} - \tfrac13\arccos\big(-k_2\cos 3\theta\big)\Big]
& \text{si } \cos 3\theta < 0 .
\end{cases}
\\]

La première branche couvre le méridien de **traction**
(\\( \theta = 0 \\)), la seconde celui de **compression**
(\\( \theta = \pi/3 \\)) ; `k₁` fixe l'ouverture de la section et `k₂ ∈ [0,1]`
son écart à un cercle. Le terme en \\( J_2 \\) rend les méridiens **courbes** et
le terme en \\( I_1 \\) les incline, si bien que la section déviatorique est un
triangle arrondi qui s'ouvre vers la compression — ce qui est tout l'objet.

### Intégrée par plan sécant, avec une normale numérique

Il n'existe pas de retour fermé exploitable sur cette surface. Pire, la normale
`∂f/∂σ` demande de dériver `λ(θ)` à travers `arccos` et `J₃` — une expression
assez longue pour qu'une erreur de signe y soit invisible en relecture et ne se
manifeste que par une direction d'écoulement légèrement fausse.

Le retour passe donc par l'algorithme du **plan sécant**, qui n'a besoin que du
scalaire `f(σ)`. Partant du prédicteur, on linéarise le critère à l'itéré
courant, on en déduit un multiplicateur, on corrige, et l'on recommence :

\\[
n^{(i)} = \frac{\partial f}{\partial\sigma}\Big|_{\sigma^{(i)}}, \qquad
\Delta\lambda^{(i)} = \frac{f(\sigma^{(i)})}{n^{(i)} : D : n^{(i)}},
\\]
\\[
\sigma^{(i+1)} = \sigma^{(i)} - \Delta\lambda^{(i)}\\,D : n^{(i)},
\qquad
\varepsilon^{p\\,(i+1)} = \varepsilon^{p\\,(i)} + \Delta\lambda^{(i)}\\,n^{(i)},
\qquad
p^{(i+1)} = p^{(i)} + \Delta\lambda^{(i)},
\\]

jusqu'à \\( |f| \le \varepsilon_{\text{tol}} \\) — `f` étant normalisé par
\\( \sigma_c \\), la tolérance l'est aussi. Le schéma est **semi-implicite** : la
normale est ré-évaluée à l'itéré courant plutôt que résolue implicitement, ce
qui converge robustement sur une surface fortement courbe **sans demander de
dérivées secondes** — c'est ce qui en fait le bon choix ici.

La normale elle-même vient de **différences centrées** sur `f` :

\\[
n_i \simeq \frac{f(\sigma + h\\,e_i) - f(\sigma - h\\,e_i)}{2h},
\qquad h = 10^{-6}\\,\sigma_c .
\\]

Le critère est ainsi exact et le gradient précis à \\( O(h^2) \\). Échanger un
gradient analytique invérifiable contre un gradient numérique qui ne peut pas
être mal dérivé est le bon compromis ici.

## La tangente cohérente, et deux limites assumées

Seule von Mises garde une tangente **analytique**, parce que seule sa forme
fermée a été confrontée à une différence finie.

> La dérivation analytique de Drucker-Prager, écrite d'abord, était **fausse de
> 24 %** — plausible, et fausse. Seul l'oracle par différences finies de
> `tests/plastic_laws.rs` l'a dit. La tangente numérique qui l'a remplacée ne
> peut pas être mal dérivée, coûte douze évaluations d'une mise à jour fermée, et
> laisse la convergence de Newton quadratique.

**La tangente stockée est symétrique.** `D_alg` voyage dans le champ d'état sous
forme de triangle supérieur (`ktan_i_j`, i ≤ j) et est relue en miroir : le format
ne peut pas porter la tangente réellement **non symétrique** d'une loi non
associée. Celle de Drucker-Prager est donc symétrisée — le compromis d'ingénierie
usuel, qui coûte à Newton son taux quadratique sur cette loi et rien d'autre, et
garde symétriques tous les consommateurs en aval.

**Une tangente doublement numérique n'est précise que jusqu'à un point.** Ottosen
dérive `f` pour obtenir sa normale, puis la tangente dérive toute cette carte
itérative ; les deux échelles d'erreur se composent, pour environ 10 % d'écart à
la dérivée exacte. Newton converge quand même — il lui faut une tangente assez
bonne pour converger, pas une tangente exacte à la précision machine — et le test
annonce le chiffre plutôt que de le cacher derrière une tolérance lâche partout.

## Mise en donnée (Rust, testé)

```rust,ignore
{{#include ../../../tests/plastic_laws.rs:example}}
```

## Exemple Python

```python
model = pyrucast.Model.drucker_prager(fes, "solid")
materials = pyrucast.element_field.material_field(
    model,
    [("E", 20_000.0), ("nu", 0.2), ("alpha", 0.3), ("k", 30.0), ("psi", 0.1)],
)
strain = pyrucast.element_field.deformation(u, fes)
state = pyrucast.element_field.integrate_behavior(model, strain, materials)
k_t = pyrucast.matrix.tangent(model, materials, state)
```

La boucle de Newton reste orchestrée en Python, comme pour toute non-linéarité :
le noyau Rust fournit la mise à jour ponctuelle et `D_alg`, pas la boucle.

## Compléments

**Ce que vaut chaque test.** Un test de plasticité qui n'affirme que « la
contrainte a baissé » ne prouve rien. Chaque loi est épinglée par sa propriété
définissante : la consistance `q = σ_y + H·p` pour l'écrouissage, l'atterrissage
sur le cône et l'effondrement au sommet pour Drucker-Prager, l'atterrissage sur
la surface à quatre paramètres et l'écart traction/compression pour Ottosen. Et
**toutes** ont leur tangente confrontée à une différence centrée des forces
internes.

**Renommage.** `Model.plasticity` s'appelle désormais `Model.plasticity_perfect`,
pour que les quatre lois se nomment de la même façon. Seul le nom a changé.
