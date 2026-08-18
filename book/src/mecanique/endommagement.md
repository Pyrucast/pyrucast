# Lois d'endommagement

## Introduction

L'endommagement décrit la **perte de raideur** d'un matériau qui se fissure, sans
déformation permanente : la contrainte est la contrainte effective, dégradée.
Comme pour la plasticité, la loi est un **attribut** du même modèle
([`DamageLaw`]) — mêmes degrés de liberté, même montage incrémental, seule change
la loi qui transforme une déformation en contrainte dégradée.

| loi | variable(s) | ce qu'elle capture |
|---|---|---|
| `mazars` | un scalaire `D` | béton, deux branches mélangées |
| `damage_tc` | `d⁺`, `d⁻` | traction et compression séparées — l'effet unilatéral |
| `damage_sic_sic` | `d_1, d_2, d_3` | composite tissé, endommagement **orthotrope** |

S'y ajoute, du côté plastique, la **plasticité poreuse** de Gurson
(`Model.gurson`), où l'endommagement est une porosité qui rétrécit la surface de
charge — voir plus bas.

## Équations continues résolues

Les trois lois partagent le cadre de la **mécanique de l'endommagement continu**
en variables d'état, avec la contrainte **effective** \\( \tilde\sigma \\) comme
pivot :

- **contrainte effective** : \\( \tilde\sigma = D_{\text{el}} : \varepsilon \\),
  celle qu'aurait le matériau sain sous la même déformation ;
- **dégradation** : \\( \sigma = (\mathbb I - \mathbb D) : \tilde\sigma \\), où
  l'opérateur \\( \mathbb D \\) est un scalaire (Mazars), une paire de scalaires
  agissant sur des parts spectrales (Damage TC), ou un tenseur diagonal dans les
  axes matériau (SiC/SiC) ;
- **moteur** : un scalaire \\( \tau \\) construit sur \\( \varepsilon \\) ou
  \\( \tilde\sigma \\) — c'est lui qui décide *ce à quoi* la loi est sensible ;
- **seuil et irréversibilité** : une variable d'histoire
  \\( r = \max\big(r_0,\ \max_{t' \le t}\tau(t')\big) \\), non décroissante ;
- **adoucissement** : \\( d = \Phi(r) \\), croissante, avec \\( \Phi(r_0) = 0 \\).

Il n'y a **pas de déformation permanente** : à décharge complète, la contrainte
revient à zéro par une droite de pente \\( (1-d)E \\). C'est ce qui distingue un
endommagement d'une plasticité, et pourquoi ces lois n'ont pas de retour sur une
surface — la mise à jour est explicite, en un pas, sans itérer.

Toute la variété des trois lois tient donc dans **trois choix** : celui du
moteur \\( \tau \\), celui de la forme de \\( \Phi \\), et celui de la structure
de \\( \mathbb D \\).

## Mazars — un scalaire

Le moteur est une **déformation équivalente** construite sur les déformations
principales positives,

\\[
\tilde\varepsilon = \sqrt{\textstyle\sum_I \langle \varepsilon_I \rangle_+^2},
\qquad \kappa = \max_t \tilde\varepsilon,
\\]

qui répond à l'extension et est aveugle à la compression hydrostatique. Deux
branches, traction et compression, partagent la même forme,

\\[
D_\bullet(\kappa) = 1 - \frac{\varepsilon_{d0}(1 - A_\bullet)}{\kappa}
- \frac{A_\bullet}{\exp\big(B_\bullet(\kappa - \varepsilon_{d0})\big)},
\qquad \bullet \in \{t, c\},
\\]

et sont mélangées par des poids issus du découpage de la contrainte effective :

\\[
D = \alpha_t D_t + \alpha_c D_c,
\qquad
\alpha_t = \frac{\sum_I \langle\varepsilon^t_I\rangle_+ \langle\varepsilon_I\rangle_+}
{\tilde\varepsilon^{\\,2}},
\\]

où \\( \varepsilon^t \\) est la déformation qu'induirait la seule part **positive**
de \\( \tilde\sigma \\) (idem \\( \alpha_c \\) avec la part négative, et
\\( \alpha_t + \alpha_c = 1 \\) sur un trajet proportionnel). C'est ce mélange
qui permet à une seule loi de décrire un matériau un ordre de grandeur plus
résistant en compression. La variable d'histoire unique étant
\\( \kappa \\), l'endommagement ne guérit pas. Le détail — dont l'axisymétrie
et la condensation en contraintes planes — est en page
[Endommagement de Mazars](mazars.md).

## Damage TC — deux variables

Mélanger les deux branches en un scalaire a un coût : un matériau endommagé en
compression l'est autant en traction, et le modèle **ne peut pas** représenter une
fissure qui se **referme** et reprend de la charge.

Damage TC les garde séparées :

\\[
\sigma = (1 - d^+)\\,\tilde\sigma^+ + (1 - d^-)\\,\tilde\sigma^-
\\]

où `σ̃⁺` et `σ̃⁻` sont les parties positive et négative de la contrainte
**effective**, découpées sur ses valeurs principales. Chacune est dégradée par sa
propre variable, si bien qu'un déchargement de la traction vers la compression
retrouve la raideur en compression — l'**effet unilatéral**, ce qui rend la loi
utilisable en cyclique.

Chaque endommagement a **son** moteur et **son** histoire :

\\[
\tau^+ = \sqrt{\tilde\sigma^+ \\!:\\! \varepsilon},
\qquad
\tau^- = \sqrt{\sqrt3\\,\big|K\\,\tilde\sigma^-_{\text{oct}} + \tilde\tau^-_{\text{oct}}\big|},
\\]
\\[
r^\pm = \max\big(r_0^\pm,\ \max_t \tau^\pm\big),
\qquad
r_0^+ = \frac{f_t}{\sqrt E},
\qquad
r_0^- = \frac{f_c}{\sqrt E}.
\\]

En traction, le moteur est l'**énergie élastique** stockée par la part positive.
En compression, c'est une mesure **octaédrique** — contrainte normale
\\( \tilde\sigma^-_{\text{oct}} = \tfrac13\operatorname{tr}\tilde\sigma^- \\) et
cisaillement \\( \tilde\tau^-_{\text{oct}} \\) — qui répond au **confinement**, et
non à la seule extension. Le coefficient `K = 0,171` est la valeur usuelle,
déduite du rapport des résistances biaxiale et uniaxiale.

Les deux lois d'adoucissement sont distinctes, et c'est délibéré :

\\[
d^+ = 1 - \frac{r_0^+}{r^+}\\,
\exp\\!\Big[A_t\Big(1 - \frac{r^+}{r_0^+}\Big)\Big],
\\]
\\[
d^- = 1 - \frac{r_0^-}{r^-}(1 - A_c)
- A_c \exp\\!\Big[2\Big(1 - \frac{r^-}{r_0^-}\Big)\Big].
\\]

Adoucissement **exponentiel** en traction — une fissure est fragile, la
contrainte tombe dès le pic ; forme **durcissante puis adoucissante** en
compression — le béton s'écrase avec un plateau, il ne casse pas net. `A_t`
règle la pente de la première, `A_c` la résistance résiduelle de la seconde.

## SiC/SiC — endommagement orthotrope

Un composite SiC/SiC est une matrice de carbure de silicium renforcée par des
torons de fibres, généralement tissés. Il ne rompt ni comme un métal ni comme un
béton : la matrice fissure d'abord, dans des plans normaux aux directions de
torons, tandis que les fibres continuent de porter la charge **à travers** ces
fissures. La raideur chute donc **par direction**, et très inégalement.

Aucun endommagement scalaire ne peut exprimer cela. Cette loi porte **un
endommagement par direction matériau** :

\\[
\kappa_i = \max_t \big\langle \varepsilon^{\text{mat}}_{ii} \big\rangle_+,
\qquad
d_i = d_{\max,i}\Big(1 - e^{-(\kappa_i - \varepsilon_{0,i})/\varepsilon_{c,i}}\Big)
\quad \text{si } \kappa_i > \varepsilon_{0,i},
\\]

où \\( \varepsilon^{\text{mat}} = R^\top \varepsilon\\,R \\) est la déformation
**dans les axes matériau**. Chaque direction a son seuil de première fissuration
\\( \varepsilon_{0,i} \\), sa vitesse de saturation \\( \varepsilon_{c,i} \\) et
son plafond \\( d_{\max,i} \\). La partie positive est essentielle : une fissure
de matrice s'ouvre en **extension** et se referme en compression, si bien qu'une
direction comprimée n'est pas dégradée du tout.

La dégradation s'applique dans ces mêmes axes, terme par terme :

\\[
\sigma^{\text{mat}}_{ij} = \sqrt{1 - d_i}\\,\sqrt{1 - d_j}\\;
\tilde\sigma^{\text{mat}}_{ij},
\qquad
\sigma = R\\,\sigma^{\text{mat}}\\,R^\top .
\\]

Le produit \\( \sqrt{1-d_i}\sqrt{1-d_j} \\) garde l'opérateur **symétrique** et
dégrade un terme de couplage autant que la plus faible des deux directions qu'il
couple ; chaque cisaillement prend la paire qu'il cisaille. À \\( d_i = d_j = d \\)
on retrouve exactement le facteur scalaire \\( (1-d) \\).

### Le repère est le tissage

Les directions d'endommagement sont les **axes matériau**, fournis exactement
comme pour l'[élasticité orthotrope](orthotropie.md) — par les vecteurs `V1`,
`V2` du champ matériau. Ce n'est pas une coïncidence : pour un composite tissé
ce **sont** les directions de torons, et réutiliser le même repère donne
gratuitement les bonnes directions maille par maille sur une pièce courbe.

### Saturation, pas rupture

Chaque `d_i` sature à `d_max,i` plutôt que d'atteindre 1. C'est l'énoncé physique
que la fissuration matricielle ne prend **pas** toute la raideur : les fibres
restent, et un composite saturé porte encore le long de ses torons. Une loi
laissant l'endommagement atteindre 1 prédirait un effondrement qui n'a pas lieu.

## Gurson — la porosité comme endommagement

Un métal ductile ne rompt pas en atteignant une contrainte : il rompt parce que
des cavités germent, croissent et coalescent jusqu'à ce que les ligaments entre
elles ne portent plus. La surface de Gurson fait de la porosité `f` une variable
interne explicite, qui **rétrécit** la surface de charge :

\\[
\Phi = \left(\frac{q}{\sigma_y}\right)^2 + 2q_1 f^* \cosh\\!\left(\frac{3q_2\sigma_m}{2\sigma_y}\right) - (1 + q_3 f^{*2})
\\]

À `f = 0` cela redonne exactement von Mises. Le `cosh` rend la contraction
dépendante de la contrainte **hydrostatique** : les cavités croissent en traction
triaxiale et se referment en compression. C'est cette sensibilité à la pression
qui fait qu'une loi J2 ne peut jamais prédire une rupture ductile.

**Coalescence** — au-delà d'une porosité critique les cavités coalescent et
l'effondrement s'accélère. Tvergaard et Needleman le modélisent en donnant à la
surface une porosité *effective* \\( f^* \\), **bilinéaire** en `f` :

\\[
f^* =
\begin{cases}
f & \text{si } f \le f_c,\\\\
f_c + \big(\tfrac{1}{q_1} - f_c\big)\dfrac{f - f_c}{f_f - f_c} & \text{sinon,}
\end{cases}
\\]

qui atteint \\( 1/q_1 \\) — surface réduite à rien, \\( \Phi \equiv 0 \\) — à la
porosité de rupture \\( f_f \\). La pente au-delà de \\( f_c \\) est
l'accélération de la coalescence ; sans elle le modèle prédit une ductilité très
supérieure à la réalité.

**Croissance** — la conservation de la masse, et rien de plus :

\\[
\dot f = (1 - f)\\,\operatorname{tr}\dot\varepsilon^p .
\\]

L'écoulement plastique sur cette surface n'est donc **pas isochore** — c'est
exactement ce qui le distingue de von Mises, et ce qui rend la croissance
possible. Le retour se fait par **plan sécant** à normale numérique, comme pour
[Ottosen](lois-plastiques.md#intégrée-par-plan-sécant-avec-une-normale-numérique),
la porosité étant remise à jour à chaque itération à partir de la part
volumique de l'incrément plastique. La **germination** n'est pas modélisée :
seule la croissance depuis une porosité initiale \\( f_0 \\).

## Mise en donnée (Rust, testé)

```rust,ignore
{{#include ../../../tests/damage_laws.rs:example}}
```

## Exemple Python

```python
{{#include ../../../tests/python/test_doc_mecanique.py:damage_tc}}
```

## Compléments

**Ce que valent les tests.** Chaque loi est épinglée sur ce qu'elle fait et que
les autres ne font pas : une fissure qui se **referme** reprend toute la charge
en compression (Damage TC), l'écrasement laisse l'endommagement de traction
intact, l'étirement selon un axe de tissage n'endommage **que** cette direction
et sature à son plafond (SiC/SiC), et la porosité de Gurson croît sous traction
triaxiale sans jamais décroître.

**Une subtilité relevée en écrivant les tests.** Sous une déformation purement
compressive, les poids `α` de Mazars s'annulent et la loi ne rapporte *aucun*
endommagement — propriété connue du modèle, et la raison même pour laquelle une
variable compressive séparée vaut son coût.

**État absent ≠ état nul.** Une loi dont l'état démarre d'une constante matériau
— la porosité initiale de Gurson — doit pouvoir distinguer « pas encore d'état »
de « état à zéro ». Le premier pas passe donc un vecteur de variables **vide**,
et non un vecteur de zéros. Sans cela un métal poreux démarrerait dense et ne
s'endommagerait jamais.

**Pas de tangente cohérente** pour les lois d'endommagement : comme Mazars, elles
ne déclarent pas de `tangent_layout`, et l'opérateur d'itération reste la rigidité
élastique non endommagée. Gurson, qui est une plasticité, en a une (numérique).
