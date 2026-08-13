# Fluage et viscoplasticité

## Introduction

Une loi **indépendante du temps** plastifie *instantanément* dès que la
contrainte atteint sa surface. Une loi visqueuse, non : la contrainte peut rester
**hors** de la surface, et l'écoulement qui l'y ramène prend du temps. C'est la
sur-contrainte qui pilote la vitesse,

\\[
\dot{p} = g(\sigma, p, \ldots)
\\]

et le pas est intégré implicitement, si bien que le résultat **dépend de `dt`**.

C'est pourquoi ces lois **erronnent** en l'absence d'incrément de temps. Intégrer
une loi de fluage comme si elle était instantanée produirait un nombre plausible
et faux ; refuser est la seule réponse honnête. L'argument existe depuis toujours
dans la signature du comportement (`dt: Option<f64>`) — ce sont les premières
lois à s'en servir.

Comme les lois plastiques, ce sont des **attributs** du même modèle : mêmes
degrés de liberté, même montage incrémental, même état interne étendu au besoin.

| loi | vitesse | ce qu'elle décrit | matériau |
|---|---|---|---|
| `creep_norton` | `ṗ = (q/K)^n` | fluage secondaire (stationnaire) | `E, nu, K, n` |
| `creep_lemaitre` | `ṗ = (q/K)^N · p^(−M)` | fluage primaire, par écrouissage en déformation | `E, nu, K, N, M` |
| `creep_blackburn` | primaire saturant + secondaire | les deux stades, dépendance en `sinh` | `E, nu, A_1, alpha_1, r_1, B_s, beta_s` |
| `viscoplasticity_chaboche` | `ṗ = ⟨(J(σ−X) − R − k)/K⟩^n` | viscoplasticité cyclique | `+ k, K, n, C_1, gamma_1, b, Q` |
| `viscoplasticity_lemaitre_chaboche` | la même, sur `σ/(1−D)` | + endommagement ductile | `+ S, s, D_c` |

## Équations continues résolues

Le cadre est celui de la [plasticité](lois-plastiques.md#équations-continues-résolues),
**moins** les conditions de Kuhn-Tucker : la partition
\\( \varepsilon = \varepsilon^e + \varepsilon^{vp} \\) et l'élasticité
\\( \sigma = D : (\varepsilon - \varepsilon^{vp}) \\) sont les mêmes, mais le
multiplicateur n'est plus une inconnue de consistance — il est **donné** par une
loi de vitesse :

\\[
\dot\varepsilon^{vp} = \dot p\\,\frac{3}{2}\\,\frac{s}{q},
\qquad
\dot p = g(q, p, \mathcal V),
\qquad
q = \sqrt{3J_2}.
\\]

L'écoulement reste **déviatorique** — donc isochore — et **radial**, la direction
étant celle de von Mises. Il n'y a plus de surface à atteindre : la contrainte
peut rester *hors* du domaine, et c'est l'écart qui pilote la vitesse à laquelle
elle y revient. Sans le facteur \\( dt \\) que porte cette équation, la loi n'a
tout simplement pas de sens.

## Un seul solveur pour toutes

Le pas est intégré par la **θ-méthode avec θ = 1** (Euler implicite) : la vitesse
est évaluée à la **fin** du pas. La direction d'écoulement étant fixée par le
prédicteur, le déviateur ne fait que se contracter,

\\[
q(\Delta p) = q^{\text{tr}} - 3\mu\\,\Delta p,
\\]

et tout le pas se ramène à une équation **scalaire** en le multiplicateur :

\\[
R(\Delta p) = \Delta p - \Delta t\\;g\big(q(\Delta p),\\; p_A + \Delta p,\\;\ldots\big) = 0 .
\\]

`R` est croissante (la vitesse décroît quand la contrainte se relaxe), donc la
racine est unique dans \\( [0,\\; q^{\text{tr}}/3\mu] \\) — la borne supérieure
étant le multiplicateur qui relaxerait *tout* le déviateur.

Le solveur fait un Newton dessus, **encadré et doublé d'une dichotomie**. Ce
n'est pas de la prudence gratuite : les fonctions de vitesse sont raides (`q^n`
avec `n` jusqu'à 20 varie de plusieurs décades à l'intérieur d'un pas), et un
Newton nu y diverge aussi volontiers qu'il converge — soit vers un multiplicateur
négatif, soit vers l'infini. Une dichotomie ne peut faire ni l'un ni l'autre.

## Les trois lois de fluage

**Norton** (ou Norton-Odqvist) est le cheval de bataille du fluage stationnaire :

\\[
\dot p = \left(\frac{q}{K}\right)^{n}.
\\]

Il n'y a **aucun seuil** : toute contrainte flue, même lentement, et c'est ce qui
distingue le fluage de la plasticité. `K` est la contrainte de référence, `n` la
sensibilité — couramment 3 à 10 pour un métal, ce qui fait varier la vitesse de
plusieurs décades à l'intérieur d'un pas.

**Lemaitre** ajoute un stade primaire par **écrouissage en déformation** :

\\[
\dot p = \left(\frac{q}{K}\right)^{N} p^{-M}, \qquad M > 0 .
\\]

La déformation accumulée ralentit elle-même l'écoulement : à contrainte
constante, l'intégration donne \\( p(t) \propto t^{1/(1+M)} \\), la courbe
concave caractéristique du fluage primaire. Aucune dépendance explicite au temps
n'apparaît, et c'est précisément ce qui rend la loi utilisable sous charge
variable — une forme à écrouissage *temporel* y serait fausse. (`p` est
plancherné à une valeur minuscule pour que \\( p^{-M} \\) reste fini au premier
pas.)

**Blackburn** décrit les deux stades, avec le primaire suivi comme sa **propre**
variable :

\\[
\dot p_{\text{prim}} = r\\,\big(\varepsilon_\infty(q) - p_{\text{prim}}\big),
\qquad \varepsilon_\infty(q) = A\\,\sinh(\alpha q),
\\]
\\[
\dot p = \dot p_{\text{prim}} + B\\,\sinh(\beta q).
\\]

Le primaire approche **exponentiellement** son asymptote \\( \varepsilon_\infty \\)
puis s'éteint (\\( r \\) est la vitesse de saturation) ; le secondaire persiste
indéfiniment. La dépendance en `sinh` est ce qui permet à un seul jeu de
paramètres de couvrir plusieurs décades de contrainte, là où une loi puissance
échoue : \\( \sinh \\) est linéaire à faible contrainte et exponentiel à forte.
Implicitement, l'équation du primaire s'inverse en un pas,

\\[
p_{\text{prim}}^{B} =
\frac{p_{\text{prim}}^{A} + \Delta t\\,r\\,A\\,\sinh(\alpha q)}{1 + \Delta t\\,r}.
\\]

> La déformation primaire est suivie comme sa **propre** variable interne
> (`p_prim`), et non déduite du total. Ce n'est qu'à cette condition que la loi
> s'intègre correctement sous charge variable — toute la raison de préférer une
> forme en déformation à une forme en temps.

## Chaboche, et sa variante endommageable

\\[
f = J(\sigma - X) - R - k, \qquad \dot{p} = \left\langle \frac{f}{K} \right\rangle^n
\\]
\\[
\dot{X} = \tfrac{2}{3}C\\,\dot{\varepsilon}_{vp} - \gamma X \dot{p},
\qquad \dot{R} = b(Q - R)\dot{p}
\\]

La contrainte de rappel `X` est ce qui rend la loi utilisable en **cyclique** :
elle translate la surface de charge, si bien que la replastification en sens
inverse survient tôt — l'effet Bauschinger, qu'aucune loi isotrope ne peut
produire. `γ` est ce qui fait **saturer** la translation au lieu de la laisser
croître sans borne (`X → C/γ`).

Les trois premières lois n'en portent pas, et c'est délibéré : un fluage décrit
un régime **monotone**, où l'effet que modélise une contrainte de rappel ne se
manifeste pas. Les deux dernières en portent, et cela leur coûte sept variables
internes de plus.

Les crochets \\( \langle\cdot\rangle \\) sont la partie positive : sous la
surface, la vitesse est nulle et le comportement redevient élastique. C'est la
seule des cinq lois à porter un **seuil** `k` — un fluage n'en a pas.

**La variante endommageable** (Lemaitre-Chaboche) remplace partout la contrainte
par la contrainte **effective**, au sens de la contrainte portée par la section
restante :

\\[
\tilde\sigma = \frac{\sigma}{1 - D},
\qquad
\dot D = \left(\frac{Y}{S}\right)^{s}\dot p,
\qquad
Y = \frac{\tilde\sigma_{\text{eq}}^{2}}{2E},
\\]

où \\( Y \\) est le **taux de restitution d'énergie élastique**, la force
thermodynamique conjuguée de `D` — sa forme complète porte un facteur de
triaxialité, pris ici à sa valeur déviatorique, simplification usuelle pour un
trajet proportionnel. Un matériau endommagé flue plus vite, ce qui l'endommage
davantage : c'est ce couplage qui produit le fluage **tertiaire** et, à `D_c`, la
rupture. `D` ne décroît jamais et est écrêté à `D_c`.

### L'intégration

La direction d'écoulement est **gelée au prédicteur**, ce qui rend le pas radial
dans l'espace décalé et le ramène à la même équation scalaire que les fluages.
Les deux variables d'écrouissage sont alors implicites en `Δp` :

\\[
X_B = \frac{X_A + \tfrac23 C\\,\Delta p\\,\hat n}{1 + \gamma\\,\Delta p},
\qquad
R_B = \frac{R_A + b\\,Q\\,\Delta p}{1 + b\\,\Delta p},
\\]

où \\( \hat n = \tfrac32 (s - X)/J \\) est la direction gelée. Les deux sont
l'inversion **exacte** des lois d'évolution discrétisées en Euler implicite, et
l'on y voit apparaître directement les asymptotes :
\\( J(X) \to C/\gamma \\) et \\( R \to Q \\) quand \\( \Delta p \to \infty \\).

Un traitement pleinement implicite ré-évaluerait la direction, au prix d'un
Newton tensoriel ; le geler est le schéma semi-implicite usuel, d'erreur du
second ordre en le pas.

> `J(σ̃ − X)` en fin de pas est calculé **sur les tenseurs**, non réduit à une
> formule scalaire. La réduction est faisable mais délicate — la contrainte de
> rappel en début de pas n'est pas parallèle à la direction d'écoulement — et
> une erreur y serait invisible. Construire le tenseur ne peut pas l'être.

## Mise en donnée (Rust, testé)

```rust,ignore
{{#include ../../../tests/viscous_laws.rs:example}}
```

## Exemple Python

```python
model = pyrucast.Model.creep_norton(fes, "solid")
materials = pyrucast.element_field.material_field(
    model, [("E", 150_000.0), ("nu", 0.3), ("K", 400.0), ("n", 5.0)]
)

# Le pas de temps est obligatoire : sans lui la loi refuse d'intégrer.
strain = pyrucast.element_field.deformation(u, fes)
state = pyrucast.element_field.integrate_behavior(model, strain, materials, dt=1e-3)

# La sortie devient le `prev` du pas suivant.
state = pyrucast.element_field.integrate_behavior(
    model, strain, materials, prev=state, dt=1e-3
)
```

## Compléments

**Ce que valent les tests.** Une loi visqueuse ne se contrôle pas sur une valeur
de contrainte : elle se contrôle **sur le temps**. D'où des tests qui vérifient
que Norton suit sa loi de vitesse en forme fermée sur un pas court, qu'une
déformation maintenue **relaxe** d'autant plus que le pas est long, que le taux
de Lemaitre **décroît** avec la déformation accumulée, que le primaire de
Blackburn **sature**, que la contrainte de rappel de Chaboche s'établit puis
plafonne sous `C/γ`, et que l'endommagement croît sans jamais guérir ni dépasser
`D_c`. Et que toutes **refusent** d'intégrer sans `dt`.

**Tangente.** Aucune de ces lois n'a de tangente analytique : toutes l'obtiennent
**par perturbation**, comme décrit au chapitre
[Lois d'écoulement plastique](lois-plastiques.md#la-tangente-cohérente-et-deux-limites-assumées).

**Ce qui n'est pas couvert.** Une seule contrainte de rappel (la version de base
de Chaboche ; deux en doubleraient l'état), et pas de dépendance à la
température — les paramètres sont des composantes matériau, donc ils peuvent
varier dans l'espace, mais ils ne sont pas fonction du champ thermique.
