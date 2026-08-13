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
- [Élasticité orthotrope et anisotrope](mecanique/orthotropie.md) — la symétrie
  matériau, repère donné par vecteurs.
- [Pression suiveuse](mecanique/pression-suiveuse.md) — une charge dont la
  direction tourne avec la surface.
- [Plasticité parfaite (von Mises)](mecanique/plasticite.md) — J2 sans
  écrouissage, retour radial (état interne `εᵖ`, `p`).
- [Lois d'écoulement plastique](mecanique/lois-plastiques.md) — la loi comme
  attribut : écrouissage isotrope, Drucker-Prager, Ottosen.
- [Fluage et viscoplasticité](mecanique/fluage.md) — les lois dépendantes du
  temps, qui exigent `dt`.
- [Endommagement de Mazars](mecanique/mazars.md) — endommagement isotrope du
  béton, deux variables (état interne `κ`).
- [Lois d'endommagement](mecanique/endommagement.md) — la loi comme attribut :
  Damage TC, SiC/SiC orthotrope, Gurson.
- [Poutre d'Euler-Bernoulli](mecanique/bernoulli.md) — sans cisaillement
  transverse, interpolation d'Hermite (1-D / plan / spatial).
- [Poutre de Timoshenko](mecanique/timoshenko.md) — flexion + cisaillement,
  intégration réduite (anti-verrouillage).
- [Portique 2D](mecanique/timoshenko.md) — poutre orientée (axial + flexion +
  cisaillement), transformation local→global.
- [Coques](mecanique/coques.md) — surface à six DDL par nœud, Reissner-Mindlin
  à cisaillement sous-intégré.
- [Cadre 3D](mecanique/timoshenko.md) — space frame 6 DOF/nœud (axial + torsion +
  flexion 2 plans), orientation automatique.
