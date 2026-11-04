# Drucker-Prager et Ottosen

## En bref

Deux lois plastiques pour les matériaux plus résistants en compression qu'en
traction : sols, roches, poudres, béton. Leur surface voit la pression, que
von Mises ignore.

| | |
|---|---|
| opérateurs | `pyrucast.model.drucker_prager(fespace, kinematics, strain_measure="linear")`, `pyrucast.model.ottosen(…)` |
| tag | `mechanical` |
| type | domaine avec comportement (`label()` : `"Plasticity"`) |
| éléments | ceux de l'[élasticité](elasticite.md) ; toutes les cinématiques |

Elles suivent le cadre de la [plasticité de von Mises](plasticite.md#théorie) :
partition de la déformation, critère \\(f \le 0\\), écoulement et conditions de
Kuhn-Tucker. Elles en diffèrent par la surface \\(f\\), le potentiel \\(g\\)
de l'écoulement (\\(\dot\varepsilon^p = \dot\lambda\\,\partial g/\partial\sigma\\)),
et la tangente.

## Théorie

Les surfaces s'écrivent sur les invariants de la contrainte :

\\[
I_1 = \operatorname{tr}\sigma, \qquad
J_2 = \tfrac12\\,s : s, \qquad
J_3 = \det s, \qquad
q = \sqrt{3J_2}, \qquad
\cos 3\theta = \frac{3\sqrt3}{2}\\,\frac{J_3}{J_2^{3/2}} ,
\\]

\\(\theta \in [0, \pi/3]\\) étant l'angle de Lode, la direction de la
contrainte dans le plan déviatorique. Von Mises ne voit que \\(q\\) ;
Drucker-Prager ajoute \\(I_1\\), donc la pression ; Ottosen ajoute
\\(\theta\\).

### Drucker-Prager

Le cône

\\[
f(\sigma) = \alpha\\,I_1 + \beta\\,q - k
\\]

a pour pente \\(\alpha\\) (`friction`), pour cohésion \\(k\\) et pour poids
déviatorique \\(\beta\\) (`beta`, 1 par défaut). L'écoulement n'est **pas
associé** : le potentiel a sa propre pente, la dilatance \\(\psi\\) (`psi`),
et son propre poids déviatorique \\(\delta\\) (`delta`, 1 par défaut),
\\(g(\sigma) = \psi\\,I_1 + \delta\\,q\\). \\(\psi = \alpha\\) redonne un
écoulement associé, \\(\psi = 0\\) un écoulement sans variation de volume.

Sur le flanc du cône, \\(q\\) et \\(I_1\\) sont affines en le multiplicateur,
et le retour se résout sans itérer. Un retour qui donnerait \\(q_B < 0\\) a
dépassé l'axe hydrostatique : la contrainte se pose alors au **sommet** du
cône, \\(s_B = 0\\), \\(\sigma_{m,B} = k/(3\alpha)\\), où la tangente est nulle :
un corps entièrement à son sommet ne porte plus rien.

**Écrouissage.** La cohésion croît avec la déformation plastique cumulée,
\\(dk = H\\,dp\\) (`H`, 0 par défaut ; négatif, il adoucit), jusqu'à une
surface **ultime** \\(\alpha_u I_1 + \beta_u q = k_u\\) (`friction_ult`,
`beta_ult`, `k_ult`, par défaut égales aux valeurs initiales). La surface
courante interpole les deux :

\\[
\lambda = \operatorname{clamp}\\!\Big(\frac{H\\,p}{k_u - k}, 0, 1\Big), \qquad
(\alpha, \beta, k)(p) = (1-\lambda)\\,(\alpha, \beta, k) + \lambda\\,(\alpha_u, \beta_u, k_u),
\\]

si bien que \\(k(p) = k + H\\,p\\) tant que la surface ultime n'est pas
atteinte. La consistance n'est alors plus linéaire : elle est résolue par un
Newton encadré, dont la solution sans écrouissage est le premier itéré. Sans
écrouissage, le retour reste fermé.

### Ottosen

Une surface à quatre paramètres, faite pour le béton :

\\[
f(\sigma) = a\\,\frac{J_2}{\sigma_c^2} + \lambda(\theta)\\,\frac{\sqrt{J_2}}{\sigma_c} + b\\,\frac{I_1}{\sigma_c} - 1,
\\]

\\[
\lambda(\theta) =
\begin{cases}
k_1 \cos\\!\big[\tfrac13 \arccos(k_2 \cos 3\theta)\big] & \text{si } \cos 3\theta \ge 0,\\\\
k_1 \cos\\!\big[\tfrac\pi3 - \tfrac13 \arccos(-k_2 \cos 3\theta)\big] & \text{si } \cos 3\theta < 0 .
\end{cases}
\\]

\\(\sigma_c\\) est la résistance en compression simple. Le terme en \\(J_2\\)
courbe les méridiens, le terme en \\(I_1\\) les incline, et \\(\lambda(\theta)\\)
fait de la section déviatorique un triangle arrondi qui s'ouvre vers la
compression.

Aucun retour fermé n'existe. Le retour passe par l'algorithme du **plan
sécant**, qui n'a besoin que de \\(f\\) : partant du prédicteur, on linéarise
le critère à l'itéré courant, on en tire un multiplicateur, on corrige, jusqu'à
\\(|f|\\) sous une tolérance. La normale \\(\partial f/\partial\sigma\\) est
obtenue par différences centrées sur \\(f\\), avec un pas de
\\(10^{-6}\\,\sigma_c\\).

### La tangente par perturbation

Les deux lois obtiennent leur tangente par **différences centrées** sur le
retour lui-même, une colonne par composante de déformation :

\\[
D_{ij} \simeq \frac{\sigma_i(\varepsilon + h\\,e_j) - \sigma_i(\varepsilon - h\\,e_j)}{2h},
\qquad h = 10^{-6}\\,\lVert\varepsilon\rVert_\infty ,
\\]

soit douze retours par point de Gauss. Les colonnes de cisaillement sont
divisées par deux, pour passer de \\(\partial\sigma/\partial\varepsilon_{ij}\\)
à \\(\partial\sigma/\partial\gamma_{ij}\\).

La tangente est **symétrisée** : seul son triangle supérieur est gardé. Celle
d'un écoulement non associé, comme Drucker-Prager, ne l'est pas ; la
symétriser coûte à Newton sa convergence quadratique sur cette loi. Celle
d'Ottosen est doublement numérique (la normale, puis la tangente) : moins
précise, elle suffit à faire converger Newton.

## Variables

| | |
|---|---|
| primales, duales | celles de l'[élasticité](elasticite.md#variables) |
| matériau requis | Drucker-Prager : `E`, `nu`, `friction`, `k`, `psi` ; Ottosen : `E`, `nu`, `a`, `b`, `k_1`, `k_2`, `sigma_c` |
| matériau facultatif | `rho`, `alpha` ; Drucker-Prager : `beta`, `delta`, `H`, `friction_ult`, `beta_ult`, `k_ult` |
| entrée, sortie, état | ceux de la [plasticité de von Mises](plasticite.md#variables) |

## Ce qu'elles produisent

| | |
|---|---|
| raideur | élastique |
| masse | exige `rho` |
| géométrique | sous la contrainte `sigma_*` |
| tangente | par perturbation, symétrisée |
| forces internes | \\(\int B^{\mathsf T}\sigma\\) |

## Exemple

La tangente de Drucker-Prager, assemblée sur une maille plastifiée :

```python
{{#include ../../../tests/python/test_doc_mecanique.py:drucker_prager}}
```

## Côté Rust

Les deux lois vivent dans `src/models/plasticity/drucker_prager.rs` et
`src/models/plasticity/ottosen.rs`, et déclarent leur tangente par
`TangentSource::Numerical`. `tests/plastic_laws.rs` vérifie que le retour
atterrit sur la surface, l'effondrement au sommet du cône, l'écart entre
traction et compression d'Ottosen, et confronte chaque tangente à une
différence finie des forces internes.
