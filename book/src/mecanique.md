# Mécanique

Les physiques mécaniques, chacune décrite sur sa page : **équations résolues**,
exemple de **mise en donnée Rust testé**, et **exemple Python**.

Comme pour la [thermique](thermique.md), ce sont des variantes de
[`SubModel`](model.md) : chacune déclare ses variables, son matériau, et
assemble sa rigidité `K` (et, le cas échéant, son comportement `COMP`).

Convention de nommage : **primal** = déplacements `u_x, u_y, u_z` (les
inconnues) ; **dual** = forces nodales `f_x, f_y, f_z` (second membre /
réactions).

- [Barre / treillis](mecanique/truss.md) — élément à effort axial (1-D/2-D/3-D).
- [Élasticité linéaire](mecanique/elasticite.md) — continuum 2-D (CP/DP) et 3-D.
- [Poutre de Timoshenko](mecanique/timoshenko.md) — flexion + cisaillement,
  intégration réduite (anti-verrouillage).
- [Portique 2D](mecanique/portique.md) — poutre orientée (axial + flexion +
  cisaillement), transformation local→global.
