# Sommaire

- [Introduction](introduction.md)
- [Correspondance Rust ↔ Python](correspondance-rust-python.md)
- [Installation et démarrage rapide](installation.md)
- [Aspect informatique](aspect-informatique.md)

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
  - [Mécanique](mecanique.md)
    - [Barre / treillis](mecanique/truss.md)
    - [Élasticité linéaire](mecanique/elasticite.md)
    - [Plasticité parfaite (von Mises)](mecanique/plasticite.md)
    - [Endommagement de Mazars](mecanique/mazars.md)
    - [Poutre de Timoshenko](mecanique/timoshenko.md)
    - [Portique 2D](mecanique/portique.md)
    - [Cadre 3D](mecanique/cadre3d.md)
  - [Contraintes](contraintes.md)
    - [Dirichlet](contraintes/dirichlet.md)
    - [Multi-points (MPC)](contraintes/mpc.md)

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
