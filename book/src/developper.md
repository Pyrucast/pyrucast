# Développer

Cette partie s'adresse aux **contributeurs** de pyrucast. Elle décrit
l'organisation du code, les conventions à respecter, le modèle mémoire
sous-jacent, comment compiler et tester, et les deux extensions les plus
courantes (ajouter une physique, ajouter un élément fini).

- [Arborescence](developper/arborescence.md) — la carte des sources : où vit
  chaque morceau (`containers/`, `ops/`, `models/`, `py/`, `viz/`…).
- [Conventions & philosophie](conventions.md) — méthode vs fonction libre,
  erreurs, `Display`/`Debug`/`dump`, sérialisation, *Definition of Done*.
- [Modèle mémoire](memory-model.md) — le store à handles : slots, générations,
  refcount, swap disque, compactage, et les évolutions prévues.
- [Parallélisme](developper/parallelisme.md) — rayon porté *au-dessus* des
  noyaux, zéro-copie, déterminisme, et ce qui reste séquentiel.
- [Compilation et tests](compilation.md) — installation détaillée, *features*
  Cargo, génération du stub `.pyi`, script « tout-en-un », dépannage.
- [Ajouter une physique](ajouter-une-physique.md) — le coût en **O(1)
  fichier** d'une nouvelle variante de `SubModel` / `SubModelKind`.
- [Ajouter un élément fini](developper/ajouter-un-element-fini.md) — un nouveau
  `ElementType`, son interpolation et sa quadrature.
- [Interrompre une fonction](developper/interrompre-une-fonction.md) — le jeton
  `Cancel` : un `Ctrl+C` ou un *timeout* qui arrête une boucle longue.

La feuille de route (décisions d'architecture verrouillées, état des lieux,
pistes futures) est dans `ROADMAP.md` à la racine du dépôt.
