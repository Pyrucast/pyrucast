# Documentation et tests

Cette page dit **quel type d'exemple vit où**, ce que chaque type de test
prouve, et surtout **qui le vérifie**. Les règles seules, sans la narration,
sont dans [`CONVENTIONS.md`](https://github.com/Pyrucast/pyrucast/blob/master/CONVENTIONS.md)
à la racine du dépôt.

## Pourquoi une page entière

Parce que le projet s'est fait prendre trois fois, de la même façon.

**Une doc qui a menti pendant tout un redécoupage.** Le remaniement d'API du
3 août 2026 a renommé `ops::assemble` en `ops::matrix`, `ops::build` en
`ops::element_field`, `ops::mesher` en `ops::mesh`. Trois pages du book
(`model.md`, `matrix.md`, `thermique.md`) sont restées sur les anciens noms.
La suite de tests était verte, la CI aussi, et personne ne l'a vu pendant des
semaines — parce que rien, nulle part, ne lisait ces pages.

**Un vérificateur qui ne vérifiait rien.** `mdbook test` tourne à chaque
`check`, égrène 85 chapitres et affiche un verdict. Il teste **zéro bloc** : les
73 clôtures Rust du book sont toutes marquées `rust,ignore`, et `ignore` veut
dire exactement « ne pas compiler ». Le pas était vert parce qu'il ne regardait
rien.

**Un garde-fou aveugle, deux fois de suite.** L'outil écrit pour compenser est
passé au vert alors qu'il ne vérifiait rien : d'abord parce que le fichier qu'il
produisait n'était pas analysable — et un fichier qui ne parse pas est vérifié
pour *rien* —, ensuite parce qu'il filtrait sur une liste de codes d'erreur
fatals, ce qui laissait passer les erreurs de syntaxe, qui n'ont pas de code.
Dans les deux cas il fallait **casser volontairement** ce qu'il surveillait pour
s'en apercevoir.

D'où la ligne directrice de toute cette page :

> **Un pas de vérification qui passe au vert sans rien regarder coûte plus cher
> que son absence** : il se lit comme de la couverture.

## Les types de test

| type | où | ce qu'il prouve | lancé par |
|---|---|---|---|
| **test unitaire** | `#[cfg(test)] mod tests` dans `src/**.rs` | le comportement d'une unité, publique ou privée | `cargo test` — [`check_rust`](../compilation.md#scriptcheck_sh--vérifications-en-bloc-ou-à-la-carte) |
| **doctest** | commentaires `///` et `//!` dans `src/**.rs` | que l'exemple **documentant un item** compile et tourne | `cargo test --doc` — `check_rust` |
| **test d'intégration** | `tests/*.rs` | une chaîne complète vue **de l'extérieur du crate** | `cargo test` — `check_rust` |
| **test Python** | `tests/python/*.py` | la surface pyo3 et le comportement côté Python | `pytest` — `check_python` |
| **garde-fou** | `tests/python/test_method_exposure.py`, `test_mirror_completeness.py` | une **invariante d'API**, pas un comportement | `pytest` — `check_python` |
| **exemple** | `examples/*.py` | une chaîne utilisateur de bout en bout | `run_examples` — `check_examples` |
| **script de formation** | `formation/*.py` | un parcours pédagogique complet | `run_examples` — `check_examples` |
| **banc** | `benches/*.rs` | une **performance**, jamais une correction | `cargo bench`, hors CI |

Trois distinctions valent d'être tenues :

- **Un test unitaire n'est pas un test d'intégration.** Le premier a accès au
  privé et teste une unité ; le second voit le crate comme un utilisateur, par
  son API publique seulement. Un comportement qui n'est prouvé qu'en unitaire
  peut être inatteignable de l'extérieur.
- **Un garde-fou n'est pas un test.** Il ne vérifie aucun calcul : il vérifie
  qu'une *règle* tient encore — que tout opérateur Rust a son binding Python,
  que tout verbe éligible a sa méthode. Il échoue quand quelqu'un a oublié une
  projection, pas quand un résultat est faux.
- **Un banc ne remplace jamais un test.** Il mesure, il n'affirme rien, et il ne
  tourne pas en CI. Le dimensionner autour de **0,5 s par itération** : en
  dessous, le bruit de mesure atteint ±18 % et noie le signal ; au-dessus, la
  mémoire s'envole. Et prendre la référence en basculant de commit avec `git`,
  pas de mémoire — puis relancer, pour trier ce qui est du bruit de ce qui est
  un écart.

## Où vit un exemple

C'est l'arbre de décision pratique. On part de **ce que l'exemple illustre**,
jamais de la page où on veut l'afficher.

| l'exemple illustre… | il vit… | le book le montre par… | vérifié par |
|---|---|---|---|
| un item de l'API Rust | un **doctest** sur l'item | rien — il est dans la rustdoc | `cargo test --doc` |
| une chaîne Rust complète (une physique) | `tests/<sujet>.rs`, entre ancres | un `include` ancré | `cargo test` |
| une chaîne Python complète | `examples/<sujet>.py` | un `include`, entier ou ancré | `run_examples` |
| un parcours pédagogique | `formation/<sujet>.py`, entre ancres | un `include` ancré | `run_examples` |
| l'usage d'un opérateur en Python | `tests/python/doc_<famille>.py`, entre ancres | un `include` ancré | `pytest` |
| une **signature de conception** | la page elle-même | ` ```rust,ignore ` — page **déclarée** | rien, et c'est assumé |

### Doctest et bloc du book ne se confondent pas

C'est la première question qu'on se pose, et la réponse est : ce sont deux
choses différentes, on ne cherche pas à les unifier.

Le **doctest** sert le lecteur de la rustdoc. Il est *à côté* de la fonction, il
répond à « comment j'appelle celle-ci ? », il tient en cinq lignes, et son
montage se cache derrière `# `. Il suit l'item : si la signature change, il
casse.

Le **bloc du book** sert le lecteur du chapitre. Il répond à « comment je monte
un calcul thermique ? », il fait quarante lignes, et il n'a de sens que dans
l'ordre du récit. Il vit dans un test d'intégration, où il est exécuté pour de
vrai.

Deux publics, deux sources. Une chaîne complète recopiée en doctest serait
illisible ; un doctest promu en chapitre ne raconterait rien.

### `include` ou `rustdoc_include`

Le mécanisme d'inclusion se choisit sur une seule question : **est-ce que
quelque chose compile déjà ce fichier ?**

- **Oui** (`tests/`, `examples/`, `formation/`, `tests/python/`, `src/`) →
  l'inclusion simple suffit. La vérification a lieu à la source ; mdbook ne fait
  que l'afficher.
- **Non** → `rustdoc_include`, qui passe le **fichier entier** à rustdoc tout en
  n'affichant que l'ancre, de sorte que `mdbook test` le compile.

Il n'y a aujourd'hui aucun fichier du second cas dans ce dépôt, et il n'y a pas
de raison d'en créer : un fichier d'exemple qui mérite d'être compilé mérite
d'être un test.

## La règle d'or

> **Aucune page du book ne possède de code.**

Toute clôture ` ```rust ` ou ` ```python ` d'une page contient une directive
d'inclusion pointant une source que la CI exécute — ou la page figure dans la
liste déclarée des **pages d'esquisses**, chacune avec sa raison écrite.

Le corollaire pratique : quand on veut ajouter un exemple à un chapitre, on
n'écrit pas dans le chapitre. On écrit un test, on l'encadre d'ancres, et on
l'inclut. L'exemple devient de la couverture au lieu d'être une dette.

État au 18 août 2026 — la règle n'est pas encore tenue partout :

| surface | conforme | écrit à la main |
|---|---|---|
| blocs Rust du book | 22 | 51 |
| blocs Python du book | 61 | 129 |

La [mise en conformité](#ce-qui-reste-à-faire) est en cours, page par page.

## Les doctests

**Tout item de l'API publique porte un exemple exécutable.** C'est le point 3 de
la *Definition of Done*, et c'est l'étage le moins cher de tous : dans un
doctest, le crate est dans la portée automatiquement, cargo gère l'édition de
liens, il n'y a ni chemin de bibliothèque à passer, ni ancre à maintenir. On
écrit l'exemple à côté de la fonction, et `cargo test` le vérifie.

Le vocabulaire des attributs, et lequel choisir :

| attribut | effet | quand |
|---|---|---|
| *(aucun)* | compile **et exécute** | le cas normal, à viser toujours |
| `no_run` | compile, n'exécute pas | ouvre une fenêtre, dure une minute, écrit un fichier |
| `compile_fail` | doit **échouer** à compiler | documenter ce que le typage interdit |
| `should_panic` | doit paniquer | documenter une précondition |
| `ignore` | **ne fait rien** | ❌ proscrit |

`ignore` est proscrit sans exception : c'est le marqueur qui ne vérifie rien, et
c'est précisément lui qui a désarmé le book 73 fois. S'il paraît nécessaire,
c'est que `no_run` était le bon choix, ou que l'exemple appartient à un test.

Deux points utiles :

- les lignes de montage se cachent avec `# ` en tête. Elles n'apparaissent pas
  dans la rustdoc, mais **elles sont compilées** — c'est ce qui permet à un
  exemple de trois lignes lisibles de reposer sur dix lignes de `Coords`, de
  nœuds et de maillage ;
- les doctests s'exécutent **aussi sur les items privés**. La visibilité n'est
  donc jamais une raison de s'en passer.

## Ce qui est vérifié, et par quoi

| règle | garde-fou | où |
|---|---|---|
| tout opérateur Rust a son binding Python | `test_mirror_completeness.py` | `check_python` |
| tout verbe éligible a sa méthode | `test_method_exposure.py` | `check_python` |
| les exemples et la formation tournent | `run_examples` | `check_examples` |
| les doctests compilent et tournent | `cargo test --doc` | `check_rust` |
| la rustdoc n'a aucun lien cassé | `cargo doc` avec `RUSTDOCFLAGS="-D warnings"` | `check_doc` |
| **les includes du book résolvent** | *à écrire* | `check_doc` |
| **aucune page ne possède de code** | *à écrire* | `check_doc` |
| **tout item public a un exemple** | *à écrire* | `check_rust` |
| **la prose ne cite pas de symbole disparu** | *à écrire* | `check_doc` |

Les quatre dernières lignes sont volontairement affichées comme manquantes.
Énoncer une règle sans dire qui la vérifie, c'est reproduire exactement le
régime qui a produit les trois pannes du début.

## Ce qui n'est pas vérifié, et pourquoi

Deux zones sont hors d'atteinte, et il vaut mieux les nommer que laisser croire
à une couverture totale.

**Les esquisses de conception.** Les pages [Ajouter une
physique](../ajouter-une-physique.md), [Modèle mémoire](../memory-model.md) et
[Interrompre une fonction](interrompre-une-fonction.md) montrent des signatures
de traits et d'énumérations avec des `/* … */`. Ce ne sont pas des exemples : ce
sont des schémas. Ils restent `rust,ignore`, ils figurent dans la liste déclarée
avec leur raison, et ils doivent **ressembler** à des esquisses — pas à du code
qu'on pourrait copier.

**La prose.** Un paragraphe qui cite `ops::matrix::stiffness` en texte courant
n'est couvert par aucun include. C'est là qu'étaient trois des erreurs trouvées
en août 2026. Le seul filet possible est un audit qui extrait les symboles cités
et les résout contre le crate et le module Python installé — c'est le quatrième
garde-fou de la table ci-dessus.

## Installer un garde-fou

Une règle de méthode, née de deux échecs consécutifs :

> **Un garde-fou n'est installé qu'une fois cassé volontairement**, au moins une
> fois, **code de retour vérifié** — pas seulement l'affichage.

Concrètement : on renomme une méthode dans la source qu'il surveille, on lance
le garde-fou, et on vérifie qu'il sort en échec. Puis on remet en état. Sans
cette étape, on ne sait pas si le vert signifie « tout va bien » ou « je ne
regarde rien ».

Deux pièges mesurés dans ce dépôt, qui donnent la mesure du risque :

- une **ancre d'inclusion inexistante** produit un bloc de code **vide**, un code
  de retour **0**, et **aucun message** : la page perd son exemple en silence ;
- un **fichier inclus absent** ne produit qu'un `[ERROR]` dans le log, avec un
  code de retour **0** lui aussi.

Autrement dit, le mécanisme sur lequel repose toute la stratégie n'est, à ce
jour, gardé par rien.

Et une règle de forme : **toute dérogation porte une raison**. Page d'esquisses,
item sans exemple, exclusion d'un garde-fou — chacune vit dans un dictionnaire
nom → raison, accompagné d'un test d'hygiène qui échoue si l'entrée devient
périmée. C'est le motif déjà en place dans `test_method_exposure.py` et
`test_mirror_completeness.py` ; il n'en est pas créé d'autre.

## Ce qui reste à faire

1. Écrire les quatre garde-fous marqués *à écrire* ci-dessus.
2. Mesurer la surface publique réelle au sens de rustdoc, pour dimensionner le
   chantier des doctests.
3. Migrer les 180 blocs écrits à la main, Python d'abord — les pages
   d'opérateurs en concentrent la moitié.
4. Retirer le filet provisoire (`cargo run --bin book_blocks`) et le pas
   `mdbook test`, qui ne teste rien, une fois la migration faite.
