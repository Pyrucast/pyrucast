# Plasticité de von Mises

## En bref

Un métal qui se déforme de façon irréversible au-delà d'une limite
d'élasticité. Deux lois, au critère de von Mises :

| | |
|---|---|
| opérateurs | `pyrucast.model.plasticity_perfect(fespace, kinematics, strain_measure="linear")` (sans écrouissage), `pyrucast.model.plasticity_isotropic(…)` (écrouissage isotrope linéaire) |
| tag | `mechanical` |
| type | domaine avec comportement (`label()` : `"Plasticity"`) |
| éléments | ceux de l'[élasticité](elasticite.md) ; toutes les cinématiques |

## Théorie

L'élastoplasticité en petites déformations partage la déformation en une part
élastique et une part plastique :

\\[
\varepsilon = \varepsilon^e + \varepsilon^p, \qquad
\sigma = D : (\varepsilon - \varepsilon^p), \qquad
f(\sigma, p) \le 0, \qquad
\dot\varepsilon^p = \dot\lambda\\,\frac{\partial f}{\partial\sigma},
\\]

avec les conditions de Kuhn-Tucker \\(\dot\lambda \ge 0\\), \\(f \le 0\\),
\\(\dot\lambda\\,f = 0\\). Le critère de von Mises porte sur la contrainte
équivalente \\(q = \sqrt{\tfrac32\\,s : s}\\), où \\(s\\) est le déviateur de
\\(\sigma\\) :

\\[
f(\sigma, p) = q - \big(\sigma_y + H\\,p\big),
\qquad \dot p = \dot\lambda ,
\\]

\\(p\\) étant la déformation plastique cumulée et \\(H\\) le module
d'écrouissage, nul pour la loi parfaite.

**Le retour radial.** Un pas part de l'état \\(A\\) (\\(\varepsilon^p_A\\),
\\(p_A\\)) et de la déformation totale \\(\varepsilon_B\\) :

1. prédiction élastique, \\(\sigma^{tr} = D : (\varepsilon_B - \varepsilon^p_A)\\),
   et sa contrainte équivalente \\(q^{tr}\\) ;
2. si \\(q^{tr} \le \sigma_y + H\\,p_A\\), le pas est élastique ;
3. sinon, la normale étant colinéaire au déviateur, la consistance est affine
   et se résout sans itérer :

\\[
\Delta p = \frac{q^{tr} - (\sigma_y + H\\,p_A)}{3\mu + H}, \qquad
s_B = s^{tr}\Big(1 - \frac{3\mu\\,\Delta p}{q^{tr}}\Big), \qquad
\varepsilon^p_B = \varepsilon^p_A + \frac{3\\,\Delta p}{2\\,q^{tr}}\\,s^{tr},
\\]

\\(\mu\\) étant le module de cisaillement ; la pression ne change pas.

Le calcul est mené en 3-D quelle que soit la cinématique. En déformations
planes, les composantes hors du plan de la déformation sont nulles ; en
contraintes planes, la condition \\(\sigma_{zz} = 0\\) est résolue par une
itération locale autour du retour radial ; en axisymétrie, la déformation
orthoradiale est mesurée.

**La tangente cohérente** est la dérivée exacte du retour,

\\[
D_\text{alg} = K\\,I\otimes I + 2\mu\\,\theta\\,\mathbb I_\text{dev} -
2\mu\\,\bar\theta\\;\hat n\otimes\hat n,
\qquad
\theta = \frac{\sigma_y + H\\,p_B}{q^{tr}},
\qquad
\bar\theta = \frac{3\mu}{3\mu + H} - (1 - \theta),
\\]

avec \\(\hat n = s^{tr}/\lVert s^{tr}\rVert\\) et \\(K\\) le module de
compressibilité. Le facteur \\(\theta\\) rend compte du pas fini : c'est ce qui
donne à Newton sa convergence quadratique.

**La raideur** reste la raideur élastique \\(\int B^{\mathsf T} D\\,B\\) : un
opérateur constant, celui d'un Newton modifié. La tangente
\\(\int B^{\mathsf T} D_\text{alg} B\\) est celle d'un Newton complet.

## Variables

En 2-D plan :

| | |
|---|---|
| primales, duales | celles de l'[élasticité](elasticite.md#variables) |
| matériau requis | `E`, `nu`, `sigma_y` ; plus `H` pour `plasticity_isotropic` |
| matériau facultatif | `rho`, `alpha` |
| entrée du comportement | `eps_xx`, `eps_yy`, `eps_xy` |
| sortie du comportement | `sigma_xx`, `sigma_yy`, `sigma_xy`, puis l'état : `eps_p_xx`…`eps_p_xy` (toujours six composantes), `p`, la déformation totale `eps_xx`…`eps_xy` (six), et `sigma_zz` |
| état relu (`prev`) | toute la sortie du pas précédent : la contrainte (avec `sigma_zz`), la déformation plastique, `p` et la déformation totale |

Au premier pas, `prev=None` part d'un état nul. Ensuite, la sortie d'un pas
se passe telle quelle comme `prev` au pas suivant.

## Ce qu'elle produit

| | |
|---|---|
| raideur | élastique |
| masse | exige `rho` |
| géométrique | sous la contrainte `sigma_*` |
| tangente | analytique, \\(D_\text{alg}\\) ; demande la déformation et `prev` |
| forces internes | \\(\int B^{\mathsf T}\sigma\\) |

## Exemple

Une maille étirée à \\(\varepsilon_{xx} = 2\cdot10^{-3}\\), au-delà de
\\(\sigma_y/E\\). La contrainte équivalente reste à \\(\sigma_y\\) :

```python
{{#include ../../../tests/python/test_doc_mecanique.py:plasticite}}
```

Avec écrouissage, elle vaut \\(\sigma_y + H\\,p\\) :

```python
{{#include ../../../tests/python/test_doc_mecanique.py:ecrouissage}}
```

La boucle de Newton complète (pas de charge, résidu
\\(F_\text{ext} - F_\text{int}\\), résolution, report de l'état) s'écrit en
Python avec ces opérateurs ; `examples/plasticite_poutre_console.py` en est un
exemple, une console cisaillée jusqu'à plastifier à l'encastrement.

## Côté Rust

`Plasticity` (`src/models/plasticity.rs`) porte la loi d'écoulement en
attribut : une même physique sert les dix lois de cette page, de la page
[Drucker-Prager et Ottosen](lois-plastiques.md), de la page
[Fluage](fluage.md) et la loi de Gurson. Les deux lois de von Mises vivent dans
`src/models/plasticity/von_mises.rs` ; `H = 0` redonne la loi parfaite par le
même chemin de code. `examples/plasticite_poutre_console.rs` est la même
console en Rust, et sert de banc de parallélisme.

```rust,ignore
{{#include ../../../tests/plastic_laws.rs:example}}
```
