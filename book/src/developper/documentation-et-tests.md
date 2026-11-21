# Documentation et tests

Cette page dit **quel type d'exemple vit où**, ce que chaque type de test
prouve, et surtout **qui le vérifie**. Les règles seules, sans la narration,
sont dans [`CONVENTIONS.md`](https://github.com/Pyrucast/pyrucast/blob/master/CONVENTIONS.md)
à la racine du dépôt.

## Le principe

Une vérification peut passer au vert sans rien regarder, et rien ne la
distingue alors d'une vérification qui réussit. Deux cas concrets :

- `mdbook test` ne compile pas un bloc marqué `rust,ignore` — `ignore` veut dire
  exactement « ne pas compiler ». Sur un book dont tous les blocs Rust sont
  inclus depuis un test, et donc marqués `ignore`, il affiche un verdict sans
  avoir rien testé ;
- un outil qui analyse un fichier qu'il a lui-même produit ne vérifie *rien* si
  ce fichier n'est pas analysable, et un filtre sur une liste de codes d'erreur
  « fatals » laisse passer les erreurs de syntaxe, qui n'ont pas de code.

D'où la ligne directrice de toute cette page :

> **Un pas de vérification qui passe au vert sans rien regarder coûte plus cher
> que son absence** : il se lit comme de la couverture.

## Les types de test

| type | où | ce qu'il prouve | lancé par |
|---|---|---|---|
| **test unitaire** | `#[cfg(test)] mod tests` dans `src/**.rs` | le comportement d'une unité, publique ou privée | `cargo test --workspace --features viz` — [`check_rust`](../compilation.md#scriptcheck_sh--vérifications-en-bloc-ou-à-la-carte) |
| **doctest** | commentaires `///` et `//!` dans `src/**.rs` | que l'exemple **documentant un item** compile et tourne | `cargo test --workspace --features viz` — `check_rust` |
| **test d'intégration** | `tests/*.rs` | une chaîne complète vue **de l'extérieur du crate** | `cargo test --workspace --features viz` — `check_rust` |
| **test Python** | `tests/python/*.py` | la surface pyo3 et le comportement côté Python | `pytest` — `check_python` |
| **garde-fou** | `tests/python/test_method_exposure.py`, `test_mirror_completeness.py`, `script/doc_lint.py` | une **règle d'API ou de documentation**, pas un comportement | `pytest` — `check_python` ; `doc_lint.py` — `check_doc` |
| **exemple** | `examples/*.py` | une chaîne utilisateur de bout en bout | `run_examples` — `check_examples` |
| **script du tutoriel** | `tutoriel/*.py` | un parcours pédagogique complet | `run_examples` — `check_examples` |
| **cas de validation** | `validation/*/*/run.py` | un cas publié (le recueil de cas tests annotés de Cast3M) tenu dans sa tolérance, comparé à la référence et à Cast3M 2026 | `pytest` (`test_validation.py`) — `check_python` |
| **banc** | `benches/*.rs` | une **performance**, jamais une correction | `cargo bench`, hors CI |

Trois distinctions valent d'être tenues :

- **Un test unitaire n'est pas un test d'intégration.** Le premier a accès au
  privé et teste une unité ; le second voit le crate comme un utilisateur, par
  son API publique seulement. Un comportement qui n'est prouvé qu'en unitaire
  peut être inatteignable de l'extérieur.
- **Un garde-fou n'est pas un test.** Il ne vérifie aucun calcul : il vérifie
  qu'une *règle* tient — que tout opérateur Rust a son binding Python, que tout
  verbe éligible a sa méthode. Il échoue quand quelqu'un a oublié une
  projection, pas quand un résultat est faux.
- **Un banc ne remplace jamais un test.** Il mesure, il n'affirme rien, et il ne
  tourne pas en CI. Le dimensionner autour de **0,5 s par itération** : en
  dessous, le bruit de mesure atteint ±18 % et noie le signal ; au-dessus, la
  mémoire s'envole. Et prendre la référence en basculant de commit avec `git`,
  pas de mémoire — puis relancer, pour trier ce qui est du bruit de ce qui est
  un écart.

Enfin, **un test qui écrit sur disque efface ce qu'il a écrit**, qu'il passe ou
non :

- côté Rust, un test d'intégration prend un `TempDir` du module `common`
  (`tests/common/mod.rs`, déclaré par `mod common;`). C'est un dossier à lui
  seul, supprimé quand la valeur tombe, y compris pendant le déroulement d'un
  test en échec ;
- côté Python, une fonction de test prend `tmp_path`, et un fichier exécuté au
  niveau module prend `tempfile.TemporaryDirectory()` (voir plus bas).

Un nom fabriqué à la main sous `std::env::temp_dir()`, ou un `mkdtemp()`, survit
au test, et les dossiers s'accumulent dans `/tmp`.

## Où vit un exemple

C'est l'arbre de décision pratique. On part de **ce que l'exemple illustre**,
jamais de la page où on veut l'afficher.

| l'exemple illustre… | il vit… | le book le montre par… | vérifié par |
|---|---|---|---|
| un item de l'API Rust | un **doctest** sur l'item | rien — il est dans la rustdoc | `cargo test` |
| une chaîne Rust complète (une physique) | `tests/<sujet>.rs`, entre ancres | un `include` ancré | `cargo test` |
| une chaîne Python complète | `examples/<sujet>.py` | un `include`, entier ou ancré | `run_examples` |
| un parcours pédagogique | `tutoriel/<sujet>.py`, entre ancres | un `include` ancré | `run_examples` |
| un cas publié (le recueil de cas tests annotés de Cast3M) | `validation/<chapitre>/<cas>/run.py` | rien — `validation/README.md` le rassemble | `pytest` (`test_validation.py`) |
| l'usage d'un opérateur en Python | `tests/python/test_doc_<famille>.py`, entre ancres ; l'ancre `module-fonction` (`mesh-skin`) le rattache à la fonction, `test_doc_reference.py` accueille ceux d'aucune page | un `include` ancré, et la référence sous la fonction | `pytest` |
| le contrat d'une fonction Python (arguments, retour, erreurs) | sa docstring, en anglais | la référence, fabriquée à partir d'elle | `doc_lint.py docstrings` |
| ce qui n'est **pas du code** (signature annotée, pseudo-code) | la page elle-même | ` ```text ` — pas de coloration, donc pas de promesse | rien, et c'est assumé |

**Le préfixe `test_` n'est pas décoratif.** `pytest` ne collecte que
`test_*.py` ; un fichier de sources d'exemples nommé `doc_ops_assemblage.py`
serait inclus dans le book et exécuté par personne. Le garde-fou `includes` ne
peut pas le voir — l'ancre résout très bien —, et la page montrerait du code
que rien ne vérifie.

### Au niveau module, pas dans une fonction

**mdbook n'enlève pas l'indentation d'un extrait inclus.** Un bloc ancré à
l'intérieur d'une fonction de test s'afficherait donc décalé de quatre espaces,
ce qu'aucun utilisateur n'écrirait. Le code ancré vit par conséquent au
**niveau module**, et le fichier se lit dans l'ordre de la page, les variables
coulant d'une ancre à la suivante, comme elles le font pour le lecteur du
chapitre.

Trois conséquences :

- pytest exécute le fichier à la **collecte**. Un exemple qui casse est une
  *erreur de collecte* et non un test en échec : le traceback est complet, la
  réécriture d'assertion fonctionne (`assert 450 == 451`), et le code de
  retour vaut 1. `--continue-on-collection-errors`, posé dans
  `pyproject.toml`, évite qu'elle interrompe le reste de la suite ;
- ces fichiers ne comptent pas de « tests » au sens de pytest. C'est une
  question d'affichage, pas de couverture : le code s'exécute et les
  assertions mordent ;
- les **fixtures ne sont pas disponibles**. Ce que `tmp_path` +
  `monkeypatch.chdir` donneraient — écrire des fichiers sous des noms courts
  sans polluer le dépôt — s'obtient en trois lignes au niveau module :
  `tempfile.TemporaryDirectory()`, un `os.chdir` en tête, et surtout un
  `os.chdir` de retour **en fin de fichier**. Omettre la restitution
  déplacerait tous les fichiers de test collectés ensuite. C'est le procédé des
  sources de `sauvegarde.md`, `installation.md` et `visualization.md`.

### Doctest et bloc du book ne se confondent pas

Ce sont deux choses différentes, et on ne cherche pas à les unifier.

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

- **Oui** (`tests/`, `examples/`, `tutoriel/`, `tests/python/`, `src/`) →
  l'inclusion simple suffit. La vérification a lieu à la source ; mdbook ne fait
  que l'afficher.
- **Non** → `rustdoc_include`, qui passe le **fichier entier** à rustdoc tout en
  n'affichant que l'ancre, de sorte que `mdbook test` le compilerait. Il
  faudrait alors ajouter `mdbook test` à `check_doc`.

Aucun fichier du dépôt n'est dans le second cas, et il n'y a pas de raison
d'en créer : un fichier d'exemple qui mérite d'être compilé mérite d'être un
test.

## La référence des opérateurs

Le contrat d'une fonction Python — ce que l'appelant doit savoir : arguments,
retour, erreurs — n'est écrit **qu'une fois**, dans sa docstring, en anglais.
Les pages de la partie *Référence* n'en contiennent pas un mot : chacune tient
en une ligne, `\{{#pymodule mesh}}`, que remplace au moment de construire le
book un petit programme, `script/book_pyapi.py`, déclaré dans `book/book.toml`
comme préprocesseur de mdBook. Il lit la docstring dans le stub commité
(`python/pyrucast/_pyrucast/__init__.pyi`), sans importer le module compilé :
la construction du site n'a rien à compiler.

Trois mots remplacés au moment de construire :

| directive | où | devient |
|---|---|---|
| `\{{#pymodule mesh}}` | une page de référence | une entrée par fonction : prototype complet, docstring, méthodes jumelles, lien vers l'algorithme, exemple |
| `\{{#pysummary mesh.line mesh.arc}}` | la page de sommaire, sous un titre de thème | un tableau : prototype abrégé et première ligne de la docstring |
| `\{{#pyalgo mesh.skin}}` | une page d'algorithme, sous le titre de la section qui raconte la fonction | la ligne « Fonctions décrites » ; la référence renvoie ici |

Seul le rangement par thème du sommaire est écrit à la main. Le garde-fou
`reference` vérifie qu'aucune fonction n'y manque, et la règle d'écriture des
docstrings est dans `CONVENTIONS.md`, partie « Docstrings of the Python API ».

**Le préprocesseur exige Python 3.** `book.toml` l'appelle par `python`. Là où
`python` désigne un Python 2 — c'est le cas hors environnement virtuel sur
certaines machines —, mdBook saute le préprocesseur **sans échouer** et la
référence sort avec ses directives brutes. On construit donc avec le Python du
venv en tête du `PATH` (`PATH="$PWD/.venv/bin:$PATH" mdbook build book`), et
le garde-fou `reference` relit le book construit pour attraper ce cas.

## La règle d'or

> **Aucune page du book ne possède de code.**

Toute clôture ` ```rust ` ou ` ```python ` d'une page contient une directive
d'inclusion pointant une source que la CI exécute. Sans exception : ce qui ne
peut pas s'exécuter n'est pas du code, et se balise ` ```text `. Le garde-fou
`fences` refuse toute clôture qui déroge à la règle.

Le corollaire pratique : quand on veut ajouter un exemple à un chapitre, on
n'écrit pas dans le chapitre. On écrit un test, on l'encadre d'ancres, et on
l'inclut. L'exemple devient de la couverture au lieu d'être une dette.

## Les doctests

**Tout item de l'API publique porte un exemple exécutable.** C'est le point 3 de
la *Definition of Done*, et c'est l'étage le moins cher de tous : dans un
doctest, le crate est dans la portée automatiquement, cargo gère l'édition de
liens, il n'y a ni chemin de bibliothèque à passer, ni ancre à maintenir. On
écrit l'exemple à côté de la fonction, et `cargo test` le vérifie.

Une exception, et une seule : **la méthode de pure délégation**. Elle expose la
face « sujet » d'un opérateur — aucune logique, un appel à la fonction libre —
et c'est cette dernière qui est la forme canonique et qui porte la
documentation. Y ajouter un exemple dupliquerait le sien, dans des fichiers
dont tout l'objet est de tenir en une ligne par verbe, et ferait un second
texte à faire vieillir.

On les reconnaît à leur **marqueur**, non à leur emplacement : toute leur
documentation tient en un « See [`module::verbe`] ». Les chercher par
répertoire ne suffirait pas : il y en a au bas de `src/ops/matrix.rs`, et
d'autres que produit une macro sur `impl $T`.

Le vocabulaire des attributs, et lequel choisir :

| attribut | effet | quand |
|---|---|---|
| *(aucun)* | compile **et exécute** | le cas normal, à viser toujours |
| `no_run` | compile, n'exécute pas | ouvre une fenêtre, dure une minute, écrit un fichier |
| `compile_fail` | doit **échouer** à compiler | documenter ce que le typage interdit |
| `should_panic` | doit paniquer | documenter une précondition |
| `ignore` | **ne fait rien** | ❌ proscrit |

`ignore` est proscrit sans exception : c'est le marqueur qui ne vérifie rien.
S'il paraît nécessaire, c'est que `no_run` était le bon choix, ou que l'exemple
appartient à un test.

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
| les exemples et le tutoriel tournent | `run_examples` | `check_examples` |
| les cas de validation tiennent leur tolérance | `test_validation.py` | `check_python` |
| les doctests compilent et tournent | `cargo test --workspace --features viz` | `check_rust` |
| la rustdoc n'a aucun lien cassé | `cargo doc` avec `RUSTDOCFLAGS="-D warnings"` | `check_doc` |
| les includes du book résolvent | `doc_lint.py includes` | `check_doc` |
| aucune page ne possède de code | `doc_lint.py fences` | `check_doc` |
| la prose ne cite pas de symbole disparu | `doc_lint.py symboles` | `check_doc` |
| tout item public a un exemple | `doc_lint.py doctests` | `check_doc` |
| toute entrée Python est citée par un exemple | `doc_lint.py api-python` | `check_doc` |
| chaque équation arrive intacte à MathJax | `doc_lint.py equations` | `check_doc` |
| toute fonction publique est dans la référence et dans le sommaire, une fois, et le book construit la contient bien | `doc_lint.py reference` | `check_doc` |
| le stub commité dit ce que dit le module installé | `doc_lint.py stub` | `check_doc` |
| une docstring suit la règle d'écriture | `doc_lint.py docstrings` | `check_doc` |

Le détail de ce que chaque script lance — commande par commande, avec les
fichiers concernés et les durées mesurées — est dans
[Compilation et tests](../compilation.md#scriptcheck_sh--vérifications-en-bloc-ou-à-la-carte).

Les neuf derniers vivent dans `script/doc_lint.py` et se lancent aussi un par
un : `python script/doc_lint.py includes`. Chacun porte son registre de
dérogations — nom → raison — et son test d'hygiène, qui échoue sur une entrée
périmée.

Quatre d'entre eux méritent une précision.

**Le registre de couverture des doctests.** Le registre
`script/doc_coverage.txt` liste les items publics qui n'ont pas d'exemple, et il
est vide. Le garde-fou `doctests` échoue dans deux cas : un item public **absent
du registre** qui n'a pas d'exemple — donc tout item nouveau —, et un item **du
registre** qui en a un sans en avoir été retiré. Le registre ne peut donc que
diminuer ; on le régénère avec `python script/doc_lint.py --ratchet`.

Le garde-fou lit la liste des items dans la rustdoc construite (`all.html`), et
n'en retient que ceux du crate : les impls **génériques** de nalgebra et
d'either, et les traits **réexportés** de rayon, y figurent sans être de l'API
de pyrucast — il les écarte en bornant l'extraction et en exigeant un lien
« source » vers `src/pyrucast/`. Deux chemins peuvent aussi désigner le même
item : rustdoc nomme le doctest d'un `impl` paramétré `CellGeom<'a>::det_j_w`,
et fait passer un **réexport** par son chemin d'origine — d'où une
normalisation des paramètres et une comparaison par sous-suite de segments.

**Le registre de citation Python** est le pendant du précédent, à une
granularité plus lâche, et le compromis mérite d'être dit. Les docstrings
Python sont écrites dans les `///` de `src/py/` et le module est compilé : le
`DocTestFinder` de la bibliothèque standard ne descend pas dans les fonctions
d'un module d'extension — sur `pyrucast.mesh` il ne trouve que le module
lui-même. Un doctest par item y demanderait donc un collecteur maison. On exige
à la place qu'une entrée publique soit **citée** par un exemple exécuté du book,
lu par AST pour qu'un nom en commentaire ou dans une chaîne ne compte pas. La
granularité est le *nom*, non l'appel : `.get(` cité une fois vaut pour les
`get` de tous les conteneurs. C'est assumé, et cela garantit ce qu'on cherchait
— aucune entrée publique nouvelle qui entre sans être montrée. Registre :
`script/python_coverage.txt`, régénéré par
`python script/doc_lint.py --ratchet-python`. Le garde-fou `docstrings` suit le
même principe, avec le registre `script/docstring_debt.txt` et
`--ratchet-docstrings`.

**Les équations** passent par Markdown avant MathJax, et Markdown les abîme
sans un mot : il mange la barre de `\{` ou de `\,`, lit une ligne `=` comme le
soulignement d'un titre et une ligne `- …` comme une puce, change une prime en
apostrophe courbe. Le garde-fou compare chaque équation du source,
échappements Markdown appliqués, au TeX que MathJax trouve dans le HTML
construit — d'où l'ordre de `check_doc`, où `mdbook build` passe avant
`doc_lint.py`. Il signale en plus, sur le source seul, les commandes dont
Markdown mange la barre et les lignes qu'il lirait comme un bloc : ces
dernières dépendent de la version de mdBook (les listes de définition
n'existent qu'à partir de la 0.5), et le HTML d'une autre version ne les
montrerait pas. Les règles d'écriture sont dans `CONVENTIONS.md`.

**L'audit de la prose** ne lit que les passages en `code inline`, hors blocs :
c'est ce qui écarte les noms de fichiers et les domaines, qui ressemblent à des
chemins. Il vérifie chaque segment, y compris le dernier — le découper en paires
laisserait `ops::matrix::stiffness` sans contrôle sur le nom qui bouge le plus
souvent. Et pour un `Type::membre`, il lit **la rustdoc**, qui seule connaît
l'appartenance réelle : un nom qui existe ailleurs dans le crate ne suffit pas à
valider la citation.

## Ce qui n'est pas vérifié, et pourquoi

Deux zones sont hors d'atteinte, et il vaut mieux les nommer que laisser croire
à une couverture totale.

**Ce qui n'est pas du code.** Une signature annotée de `/* … */`, une
énumération abrégée par `// … une ligne par physique`, un pseudo-code `f(...)` :
ces blocs se balisent ` ```text `. La coloration syntaxique mentirait, et le
lecteur voit ainsi tout de suite qu'il ne peut pas les copier. Ce n'est pas une
dérogation à la règle d'or — c'est reconnaître que ces blocs ne relèvent pas
d'elle.

La nuance se juge **bloc par bloc, jamais page par page**. Une page de principes
contient aussi du code réel — la déclaration de `Handle<T>`, celle du trait
`Cancel`, l'implémentation `PySignals` — et ces blocs-là sont inclus depuis
`src/`, comme tout autre code.

**La prose.** Un paragraphe qui cite `ops::matrix::stiffness` en texte courant
n'est couvert par aucun include. Le seul filet possible est un audit qui
extrait les symboles cités et les résout contre le crate et le module Python
installé — c'est le garde-fou `symboles`. Il ne lit que des noms : une phrase
qui décrit mal ce que fait une fonction bien nommée lui échappe.

## Installer un garde-fou

Une règle de méthode :

> **Un garde-fou n'est installé qu'une fois cassé volontairement**, au moins une
> fois, **code de retour vérifié** — pas seulement l'affichage.

Concrètement : on renomme une méthode dans la source qu'il surveille, on lance
le garde-fou, et on vérifie qu'il sort en échec. Puis on remet en état. Sans
cette étape, on ne sait pas si le vert signifie « tout va bien » ou « je ne
regarde rien ».

Deux comportements de mdBook montrent le risque, et justifient le garde-fou
`includes` :

- une **ancre d'inclusion inexistante** produit un bloc de code **vide**, un code
  de retour **0**, et **aucun message** : la page perd son exemple en silence ;
- un **fichier inclus absent** ne produit qu'un `[ERROR]` dans le log, avec un
  code de retour **0** lui aussi.

Et une règle de forme : **toute dérogation porte une raison**. Item sans
exemple, exclusion d'un garde-fou — chacune vit dans un dictionnaire ou un
registre nom → raison, accompagné d'un test d'hygiène qui échoue si l'entrée
devient périmée. C'est le motif de `test_method_exposure.py`, de
`test_mirror_completeness.py` et des registres de `doc_lint.py`.
