# Sommaire

- [Introduction](introduction.md)
- [Correspondance Rust ↔ Python](correspondance-rust-python.md)
- [Installation et démarrage rapide](installation.md)
- [Aspect informatique](aspect-informatique.md)
- [Mailler une géométrie](mailler.md)

# Objets

- [Vue d'ensemble](objets.md)
  - [Coordonnées (Coords)](coords.md)
  - [Nœud (Node)](node.md)
  - [Agrégat](aggregate.md)
  - [Maillage (Mesh / SubMesh)](mesh.md)
  - [Espace éléments finis](fe-space.md)
  - [Champ (Field / SubField)](field.md)
  - [Champ aux nœuds](node-field.md)
  - [Champ aux points de Gauss](element-field.md)
  - [Modèle physique](model.md)
  - [Matrice creuse](matrix.md)
  - [Évolution (Evolution / SubEvolution)](evolution.md)

# Détails des physiques

- [Vue d'ensemble](physiques.md)
  - [Conduction thermique](thermique.md)
  - [Diffusion (loi de Fick)](diffusion.md)
  - [Mécanique](mecanique.md)
    - [Barre / treillis](mecanique/truss.md)
    - [Élasticité linéaire](mecanique/elasticite.md)
    - [Élasticité orthotrope et anisotrope](mecanique/orthotropie.md)
    - [Pression suiveuse](mecanique/pression-suiveuse.md)
    - [Plasticité parfaite (von Mises)](mecanique/plasticite.md)
    - [Lois d'écoulement plastique](mecanique/lois-plastiques.md)
    - [Fluage et viscoplasticité](mecanique/fluage.md)
    - [Endommagement de Mazars](mecanique/mazars.md)
    - [Lois d'endommagement](mecanique/endommagement.md)
    - [Poutre de Timoshenko](mecanique/timoshenko.md)
    - [Portique 2D](mecanique/portique.md)
    - [Cadre 3D](mecanique/cadre3d.md)
  - [Contraintes](contraintes.md)
    - [Dirichlet](contraintes/dirichlet.md)
    - [Multi-points (MPC)](contraintes/mpc.md)
    - [Baignage (embedded)](contraintes/embedded.md)
    - [Contact (nœud-surface)](contraintes/contact.md)

# Éléments finis supportés

- [Catalogue](elements/index.md)
  - [SEG2 — segment linéaire](elements/seg2.md)
  - [TRI3 — triangle linéaire](elements/tri3.md)
  - [QUA4 — quadrangle bilinéaire](elements/qua4.md)
  - [TET4 — tétraèdre linéaire](elements/tet4.md)
  - [PYRA5 — pyramide linéaire](elements/pyra5.md)
  - [PENTA6 — prisme linéaire](elements/penta6.md)
  - [HEX8 — hexaèdre trilinéaire](elements/hex8.md)
  - [SEG3 — segment quadratique](elements/seg3.md)
  - [TRI6 — triangle quadratique](elements/tri6.md)
  - [QUA8 — quadrangle sérendipité](elements/qua8.md)
  - [QUA9 — quadrangle biquadratique](elements/qua9.md)
  - [TET10 — tétraèdre quadratique](elements/tet10.md)
  - [PENTA15 — prisme sérendipité](elements/penta15.md)
  - [HEX20 — hexaèdre sérendipité](elements/hex20.md)
  - [HEX27 — hexaèdre tri-quadratique](elements/hex27.md)

# Détail des opérateurs

- [Vue d'ensemble](operateurs.md)
  - [Maillage](operateurs/maillage.md)
    - [Triangulation : briques mathématiques](triangulation.md)
  - [Construction](operateurs/construction.md)
  - [Géométrie](operateurs/geometrie.md)
  - [Champs](operateurs/champs.md)
  - [Assemblage](operateurs/assemblage.md)
  - [Comportement](operateurs/comportement.md)
  - [Solveur](operateurs/solveur.md)
  - [Visualisation](visualization.md)

# Couche Python haut niveau

- [Thermo-mécanique pas-à-pas](thermomecanique-pas-a-pas.md)

# Formations

- [Formation débutant](formation/debutant.md)
  - [Présentation de pyrucast](formation/presentation.md)
  - [Python & conventions pyrucast](formation/langage-python.md)
  - [Maillage](formation/maillage.md)
  - [Calcul thermique](formation/thermique.md)
  - [Calcul mécanique](formation/mecanique.md)
  - [Compléments](formation/complements.md)

# Développer

- [Vue d'ensemble](developper.md)
  - [Arborescence](developper/arborescence.md)
  - [Conventions & philosophie](conventions.md)
  - [Modèle mémoire](memory-model.md)
  - [Parallélisme](developper/parallelisme.md)
  - [Compilation et tests](compilation.md)
  - [Ajouter une physique](ajouter-une-physique.md)
  - [Ajouter un élément fini](developper/ajouter-un-element-fini.md)
  - [Interrompre une fonction](developper/interrompre-une-fonction.md)
