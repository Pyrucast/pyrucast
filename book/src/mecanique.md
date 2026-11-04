# Mécanique

Les physiques de tag `mechanical`. Leurs inconnues sont les déplacements
`u_x`, `u_y`, `u_z` (et les rotations `r_x`, `r_y`, `r_z` des poutres et des
coques), leurs duales les forces nodales `f_x`, `f_y`, `f_z` (et les moments
`m_x`, `m_y`, `m_z`).

**Milieux continus**, une même cinématique (`B`, la déformation tirée du
déplacement) et une loi qui change :

- [Élasticité linéaire](mecanique/elasticite.md) — `elasticity`, et sa
  [symétrie matériau orthotrope ou anisotrope](mecanique/orthotropie.md) ;
- [Plasticité de von Mises](mecanique/plasticite.md) — `plasticity_perfect`,
  `plasticity_isotropic` ;
- [Drucker-Prager et Ottosen](mecanique/lois-plastiques.md) — des surfaces qui
  voient la pression ;
- [Fluage et viscoplasticité](mecanique/fluage.md) — Norton, Blackburn,
  Lemaitre, Chaboche, Lemaitre-Chaboche ;
- [Endommagement de Mazars](mecanique/mazars.md) ;
- [Autres lois d'endommagement](mecanique/endommagement.md) — Damage TC,
  SiC/SiC, Gurson ;
- [Grandes transformations](mecanique/grandes-transformations.md) — les mesures
  de déformation `green_lagrange` et `hencky`.

**Éléments de structure**, dont les efforts sont intégrés sur la section :

- [Barre](mecanique/truss.md) — `truss`, l'effort axial seul ;
- [Poutre d'Euler-Bernoulli](mecanique/bernoulli.md) — `bernoulli` ;
- [Poutre de Timoshenko](mecanique/timoshenko.md) — `timoshenko`, le
  cisaillement en plus ;
- [Coques](mecanique/coques.md) — `shell`, six inconnues par nœud.

## Ce que partagent les milieux continus

Une loi de milieu continu se déclare par
`pyrucast.model.<loi>(fespace, kinematics, …, strain_measure="linear")`.

`kinematics` est l'hypothèse cinématique :

| `kinematics` | espace | composantes de la déformation et de la contrainte |
|---|---|---|
| `"plane_stress"` | 2-D, \\(\sigma_{zz} = 0\\) | `xx`, `yy`, `xy` |
| `"plane_strain"` | 2-D, \\(\varepsilon_{zz} = 0\\) | `xx`, `yy`, `xy` |
| `"axisymmetric"` | 2-D, plan méridien d'un solide de révolution | `xx`, `yy`, `zz`, `xy`, où `zz` est l'orthoradiale \\(\theta\theta\\) |
| `"full_3d"` | 3-D | `xx`, `yy`, `zz`, `yz`, `xz`, `xy` |

L'axisymétrie demande des `Coords` de révolution
(voir [Coordonnées](coords.md#repère-de-révolution)), et réciproquement : un
modèle plan est refusé sur une géométrie de révolution. La déformation suit la
convention de l'ingénieur : le glissement \\(\gamma_{xy} = 2\varepsilon_{xy}\\)
est rangé dans `eps_xy`.

Les lois plastiques et de fluage intègrent leur état en 3-D, quelle que soit
la cinématique, et rendent en plus de la contrainte la déformation totale et,
pour les modèles plans, `sigma_zz`. La sortie d'un pas de toute loi non
linéaire est complète : elle sert telle quelle de `prev` au pas suivant.

Le matériau accepte deux composantes facultatives : `rho` (la masse) et
`alpha` (la dilatation thermique, lue par
`pyrucast.element_field.thermal_strain`).
