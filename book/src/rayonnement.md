# Rayonnement

## En bref

Le rayonnement d'une surface vers un environnement lointain à la température
\\(T_\infty\\) (loi de Stefan-Boltzmann). Il se pose sur le bord d'une
conduction et partage ses variables : il entre directement dans sa matrice.

| | |
|---|---|
| opérateur | `pyrucast.model.radiation(fespace, target)` |
| tag | `thermal` et `radiation` |
| type | domaine avec comportement (`label()` : `"Radiation"`) |
| éléments | le bord d'une conduction : `SEG2`, `SEG3` en 2-D ; `TRI3`, `QUA4`, `TRI6`, `QUA8`, `QUA9` en 3-D |

`fespace` est l'espace du bord, `target` la conduction qu'il refroidit. Une
cible qui n'assemble pas `T` avec sa duale `q` est refusée : le rayonnement
écrirait dans une ligne que personne ne résout.

Les deux tags permettent de garder le rayonnement dans le problème thermique
(tag `thermal`) ou de le prendre seul (tag `radiation`) pour l'assembler ou
l'inspecter à part.

## Théorie

Le flux qui quitte la surface vaut

\\[
q\cdot n = \sigma\\,\varepsilon\\,\big(T^4 - T_\infty^4\big),
\\]

où \\(\sigma\\) est la constante de Stefan-Boltzmann et \\(\varepsilon\\)
l'émissivité. Aucune normale n'entre dans le calcul : la direction est déjà
dans \\(q\cdot n\\), il ne reste qu'un scalaire et la mesure de la surface.

La loi n'est pas linéaire en \\(T\\), d'où trois termes :

| terme | expression |
|---|---|
| raideur | \\(4\sigma\varepsilon T_\infty^3 \int_\Gamma N_i N_j\\,d\Gamma\\) : le film linéarisé autour de \\(T_\infty\\) |
| forces internes | \\(\int_\Gamma N_i\\,\sigma\varepsilon\\,(T^4 - T_\infty^4)\\,d\Gamma\\) : le résidu exact |
| tangente | \\(4\sigma\varepsilon T^3 \int_\Gamma N_i N_j\\,d\Gamma\\), à la température courante |

Linéarisée autour de \\(T_\infty\\), la raideur est une matrice constante :
c'est l'opérateur de départ d'une boucle de Newton, et à elle seule un
opérateur de point fixe utilisable. La tangente porte la non-linéarité : elle
est évaluée à la température de chaque point de Gauss.

`sigma` vaut par défaut la constante SI, `5,670374419·10⁻⁸ W·m⁻²·K⁻⁴`, et
`T` est alors une température **absolue**, en kelvins : une puissance
quatrième ne permet pas de déplacer l'origine. Dans un autre système
d'unités, on donne `sigma` dans le matériau.

## Variables

| | |
|---|---|
| primale | `T`, celle de la conduction |
| duale | `q`, celle de la conduction |
| matériau requis | `emis` (l'émissivité), `T_inf` |
| matériau facultatif | `sigma` |
| entrée du comportement | `T`, la température aux points de Gauss du bord |
| sortie du comportement | `flux`, \\(\sigma\varepsilon(T^4 - T_\infty^4)\\) |
| état | aucun |

## Ce qu'il produit

| | |
|---|---|
| raideur | le film linéarisé |
| masse | — |
| géométrique | — |
| tangente | analytique, \\(4\sigma\varepsilon T^3\\) |
| forces internes | le flux rayonné, pondéré par \\(N_i\\) |
| forces externes | aucune |

## Dans un calcul

**Le flux se calcule aux points de Gauss de la face.** La température y est
interpolée à partir de ses valeurs aux nœuds,
\\( T\_g = \sum\_j N\_j(x\_g)\\,T\_j \\), élevée à la puissance quatrième
**en ce point**, et le flux \\( \sigma\varepsilon(T\_g^4 - T\_\infty^4) \\) est
intégré avec le poids \\( w\_g \\) de chaque point (la mesure de la surface
comprise) :

\\[ R\_i = \sum\_g w\_g\\, N\_i(x\_g)\\, \sigma\varepsilon\\,(T\_g^4 - T\_\infty^4),
\qquad (K\_r)\_{ij} = \sum\_g w\_g\\, 4\sigma\varepsilon\\,T\_g^3\\, N\_i(x\_g)\\, N\_j(x\_g) \\]

\\( R \\) est le résidu, que rendent les forces internes, et \\( K\_r \\) sa
dérivée exacte par rapport aux températures nodales : la tangente.

**Le calcul pas à pas annule le résidu par corrections successives.** À chaque
pas de temps, [`pyrucast.stepping`](pas-a-pas.md) part de la température du
pas précédent et calcule le résidu de tout le problème — conduction, capacité,
chargements, rayonnement. Il résout la correction avec un opérateur qui
contient la tangente \\( K\_r \\), prise une fois au début du pas, puis
recommence avec le résidu de la nouvelle température, jusqu'à ce que la
correction tombe sous `precision` (\\( 10^{-4} \\) par défaut) relativement
à la température. Le résidu suit donc la température à chaque itération ;
l'opérateur, lui, reste celui du début du pas. Comme il contient la dérivée de
\\( T^4 \\), une ou deux corrections suffisent quand le pas de temps est
raisonnable (algorithme détaillé dans [Calcul pas à pas](pas-a-pas.md),
Algo 2).

La raideur, film linéarisé autour de \\( T\_\infty \\), est une matrice
constante : employée seule à chaque itération, elle donne une itération de
point fixe, qui converge plus lentement que la tangente.

> **Une autre méthode** consiste à écrire le rayonnement **comme une
> convection** dont le coefficient dépend de la température. Puisque
> \\( T^4 - T\_\infty^4 = (T^2 + T\_\infty^2)(T + T\_\infty)(T - T\_\infty) \\),
> le flux vaut \\( h\_r\\,(T - T\_\infty) \\), avec
>
> \\[ h\_r = \sigma\varepsilon\\,(T^2 + T\_\infty^2)(T + T\_\infty). \\]
>
> On calcule \\( h\_r \\) **aux nœuds**, avec la température nodale de
> l'itération en cours, on l'interpole aux points de Gauss, et chaque itération
> résout un problème de convection : \\( \int\_\Gamma h\_r N\_i N\_j \\) dans
> la matrice, \\( \int\_\Gamma h\_r\\,T\_\infty N\_i \\) au second membre. On
> recommence avec la nouvelle température jusqu'à ce qu'elle ne bouge plus.
>
> Deux choses changent. \\( h\_r \\) est une **sécante**, pas la tangente
> \\( 4\sigma\varepsilon T^3 \\) : chaque itération corrige moins, et il en
> faut davantage pour la même précision. Et le problème discret n'est pas tout
> à fait le même : \\( h\_r \\) interpolé depuis les nœuds n'est pas
> \\( h\_r \\) calculé avec la température des points de Gauss. Les deux
> méthodes convergent vers la même solution quand le maillage s'affine ; sur un
> maillage donné, leurs solutions convergées diffèrent de l'ordre de l'erreur
> de discrétisation.

## Exemple

Une plaque à 1000 K qui rayonne par son bord supérieur vers 300 K. On lit le
flux au bord, puis l'écart entre tangente et raideur au coin du bord : un seul
segment, de longueur `L = 0,5`, touche ce coin, où \\(\int N_0^2 = L/3\\) :

```python
{{#include ../../tests/python/test_doc_thermique.py:rayonnement}}
```

## Côté Rust

`Radiation` (`src/models/radiation.rs`) garde son sous-espace de bord. Son
noyau de raideur évalue `4σεT∞³` au point de Gauss, son comportement
(`integrate_point`) le flux, et `tangent_point` la dérivée `4σεT³`.
`STEFAN_BOLTZMANN` est la constante par défaut.

Le test compare le flux rayonné à `σε(T⁴ − T∞⁴)` fois l'aire, et la tangente à
une différence finie du résidu :

```rust,ignore
{{#include ../../tests/radiation.rs:example}}
```
