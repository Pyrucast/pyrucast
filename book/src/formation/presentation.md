# Présentation de pyrucast

## pyrucast, quid ?

Bibliothèque **éléments finis** écrite en Rust, exposée à Python. Comme
Cast3M, elle résout des équations aux dérivées partielles par la méthode des
éléments finis — mais c'est un projet bien plus jeune et bien plus étroit :
un socle de calcul (maillage, assemblage, résolution), pas un système
complet avec pré/post-processeur graphique intégré, ni des décennies de
physiques capitalisées.

- **Résolution d'équations aux dérivées partielles**, comme Cast3M.
- **Système en couches**, pas un système fermé : maillage, éléments finis,
  assemblage, comportement, solveur et visualisation sont des modules
  séparés que l'on compose depuis Python (ou directement depuis Rust — voir
  [Installation](../installation.md)).
- **Pas de langage de commande dédié.** Là où Cast3M invente Gibiane,
  pyrucast s'utilise en **Python ordinaire** : les « opérateurs » sont des
  fonctions, les « objets » sont des classes. Voir
  [Python & conventions pyrucast](langage-python.md).

## Domaines couverts

> **Cast3M couvre plus large.** Le support Cast3M original liste la mécanique
> des structures (quasi-statique, contact, dynamique, rupture XFEM), la
> thermique (conduction/convection/rayonnement/changement de phase), la
> mécanique des fluides, la diffusion multi-espèces, la fabrication
> additive, la magnétostatique, le couplage thermo-hygro-mécanique et
> l'optimisation topologique. **pyrucast ne couvre, à ce jour, que les deux
> premiers points — et partiellement.** Le reste (fluides, magnétostatique,
> diffusion, optimisation topologique, rupture) n'existe pas dans pyrucast ;
> il n'en sera plus question dans cette formation.

Ce que pyrucast fait réellement :

- **Mécanique des structures**, quasi-statique, petites déformations :
  élasticité linéaire, plasticité parfaite de von Mises, endommagement de
  Mazars, éléments structuraux (barre, poutre de Timoshenko, portique 2D,
  cadre 3D) ; matrices de masse cohérente et de rigidité géométrique ;
  contraintes multi-points (MPC), baignage (« embedded »), **contact
  unilatéral nœud-surface**.
- **Thermique**, conduction stationnaire + convection (film/Robin).

> **Non disponible dans pyrucast.** Pas de rayonnement, pas de changement de
> phase, pas de terme transitoire (capacité) câblé dans une boucle en temps —
> chaque pas thermique résout un problème **stationnaire** (voir
> [Calcul thermique](thermique.md) pour le détail). Pas de dynamique
> (temporelle ou modale), pas de flambage, pas de rupture (XFEM). La liste
> évolue vite : elle sera obsolète avant longtemps, mais reflète l'état au
> moment de l'écriture de cette formation.

## Comment obtenir pyrucast ?

- **Multiplateforme** : Linux, macOS, Windows — tout ce qu'accepte la chaîne
  Rust + Python (`rustup`, `pip`).
- **Où le télécharger ?** Le dépôt du projet (voir le lien en tête de la page
  [Formation débutant](debutant.md)).
- **Code source** : toujours accessible — pyrucast est un projet Rust
  ordinaire, sans build fermé.
- **Prix** : logiciel libre.

## Comment utiliser pyrucast ?

1. Écrire un script Python — un fichier texte ordinaire, extension `.py`.
2. Ouvrir un terminal, se placer dans le dépôt cloné.
3. Compiler puis lancer le script :

   ```bash
   maturin develop --release
   python mon_script.py
   ```

4. Utilisable aussi en mode interactif (`python`, ou un notebook) — chaque
   `import pyrucast` recharge la même API.

Voir [Installation et démarrage rapide](../installation.md) pour le détail
(prérequis, venv, vérification).

## Où trouver la documentation ?

- Ce livre — théorie et référence des objets/opérateurs.
- La documentation de l'API Rust : `cargo doc --no-deps --lib --open`.
- Les scripts de cette formation : dossier `formation/` du dépôt.
- Des exemples plus nombreux, un par sujet : dossier `examples/` du dépôt.
