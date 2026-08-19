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

**Un vérificateur qui ne vérifiait rien.** `mdbook test` tournait à chaque
`check`, égrenait 85 chapitres et affichait un verdict. Il testait **zéro
bloc** : les 73 clôtures Rust du book étaient toutes marquées `rust,ignore`, et
`ignore` veut dire exactement « ne pas compiler ». Le pas était vert parce
qu'il ne regardait rien — il a depuis été retiré de la chaîne.

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
| l'usage d'un opérateur en Python | `tests/python/test_doc_<famille>.py`, entre ancres | un `include` ancré | `pytest` |
| ce qui n'est **pas du code** (signature annotée, pseudo-code) | la page elle-même | ` ```text ` — pas de coloration, donc pas de promesse | rien, et c'est assumé |

Un piège vérifié à la première migration : **le préfixe `test_` n'est pas
décoratif.** `pytest` ne collecte que `test_*.py` ; un fichier de sources
d'exemples nommé `doc_ops_assemblage.py` est inclus dans le book et exécuté par
personne. Le garde-fou `includes` ne peut pas le voir — l'ancre résout très
bien —, et on retombe sur une page qui montre du code que rien ne vérifie.

### Au niveau module, pas dans une fonction

**mdbook n'enlève pas l'indentation d'un extrait inclus.** Un bloc ancré à
l'intérieur d'une fonction de test s'affiche donc décalé de quatre espaces,
sur chaque page, indéfiniment — ce qu'aucun utilisateur n'écrirait. Le code
ancré vit par conséquent au **niveau module**, et le fichier se lit dans
l'ordre de la page, les variables coulant d'une ancre à la suivante, comme
elles le font pour le lecteur du chapitre.

Trois conséquences, toutes mesurées :

- pytest exécute le fichier à la **collecte**. Un exemple qui casse est une
  *erreur de collecte* et non un test en échec : le traceback est complet, la
  réécriture d'assertion fonctionne (`assert 450 == 451`), et le code de
  retour vaut 1. `--continue-on-collection-errors`, posé dans
  `pyproject.toml`, évite qu'elle interrompe le reste de la suite ;
- ces fichiers ne comptent plus de « tests » au sens de pytest. C'est un
  changement d'affichage, pas de couverture : le code s'exécute et les
  assertions mordent ;
- les **fixtures ne sont pas disponibles**. Ce que `tmp_path` +
  `monkeypatch.chdir` donneraient — écrire des fichiers sous des noms courts
  sans polluer le dépôt — s'obtient en trois lignes au niveau module :
  `tempfile.TemporaryDirectory()`, un `os.chdir` en tête, et surtout un
  `os.chdir` de retour **en fin de fichier**. Omettre la restitution
  déplacerait tous les fichiers de test collectés ensuite. C'est le procédé de
  `sauvegarde.md`, `installation.md` et `visualization.md`.

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
  n'affichant que l'ancre, de sorte que `mdbook test` le compilerait. Il
  faudrait alors le remettre dans `check_doc`, d'où il a été retiré.

Il n'y a aujourd'hui aucun fichier du second cas dans ce dépôt, et il n'y a pas
de raison d'en créer : un fichier d'exemple qui mérite d'être compilé mérite
d'être un test.

## La règle d'or

> **Aucune page du book ne possède de code.**

Toute clôture ` ```rust ` ou ` ```python ` d'une page contient une directive
d'inclusion pointant une source que la CI exécute. Sans exception : ce qui ne
peut pas s'exécuter n'est pas du code, et se balise ` ```text `.

Le corollaire pratique : quand on veut ajouter un exemple à un chapitre, on
n'écrit pas dans le chapitre. On écrit un test, on l'encadre d'ancres, et on
l'inclut. L'exemple devient de la couverture au lieu d'être une dette.

État au 18 août 2026 :

| surface | conforme | écrit à la main |
|---|---|---|
| blocs Rust du book | **69** | 0 |
| blocs Python du book | **188** | 0 |

C'était 22 sur 73 et 61 sur 190 quand ces conventions ont été écrites. Le
garde-fou `fences` empêche désormais d'en réintroduire.

## Les doctests

**Tout item de l'API publique porte un exemple exécutable.** C'est le point 3 de
la *Definition of Done*, et c'est l'étage le moins cher de tous : dans un
doctest, le crate est dans la portée automatiquement, cargo gère l'édition de
liens, il n'y a ni chemin de bibliothèque à passer, ni ancre à maintenir. On
écrit l'exemple à côté de la fonction, et `cargo test` le vérifie.

Le vocabulaire des attributs, et lequel choisir :

Une exception, et une seule : **la méthode de pure délégation**. Elle expose la
face « sujet » d'un opérateur — aucune logique, un appel à la fonction libre —
et c'est cette dernière qui est la forme canonique et qui porte la
documentation. Y ajouter un exemple dupliquerait le sien : soixante-treize
exemples de plus, plus de six cents lignes dans des fichiers dont tout l'objet
est de tenir en une ligne par verbe, et un second texte à faire vieillir.

On les reconnaît à leur **marqueur**, non à leur emplacement : toute leur
documentation tient en un « voir [`module::verbe`] ». Chercher par répertoire en
laissait passer la moitié — il y en a au bas de `src/ops/matrix.rs`, et d'autres
que produit une macro sur `impl $T`.

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
| les includes du book résolvent | `doc_lint.py includes` | `check_doc` |
| aucune page ne possède de code | `doc_lint.py fences` | `check_doc` |
| la prose ne cite pas de symbole disparu | `doc_lint.py symboles` | `check_doc` |
| tout item public a un exemple | `doc_lint.py doctests` | `check_doc` |

Les quatre derniers vivent dans `script/doc_lint.py` et se lancent aussi un par
un : `python script/doc_lint.py includes`. Chacun porte son registre de
dérogations — nom → raison — et son test d'hygiène, qui échoue sur une entrée
périmée.

Deux d'entre eux méritent une précision.

**Le cliquet de couverture.** La règle « tout item public porte un exemple » ne
pouvait pas être vérifiée frontalement au départ : la dette était de **1531
items sur 1542**, et un garde-fou rouge dès le premier jour finit désactivé. Le
registre `script/doc_coverage.txt` liste donc les items qui n'en ont pas, et le
garde-fou échoue dans deux cas : un item public **absent du registre** qui n'a
pas d'exemple — donc tout item nouveau —, et un item **du registre** qui en a
désormais un sans avoir été retiré. La liste ne pouvant que fondre, sa fonte a
servi d'indicateur d'avancement jusqu'à ce qu'elle **atteigne zéro**, le
2026-08-19. Le registre reste en place, vide : c'est lui qui refuse un item
public nouveau sans exemple. On le régénère avec
`python script/doc_lint.py --ratchet`.

Sur ce chemin, le cliquet lui-même a dû apprendre trois choses. Un item n'est
pas de nous parce qu'il figure dans `all.html` : les impls **génériques** de
nalgebra et d'either, et les traits **réexportés** de rayon, y entrent sans être
de l'API du crate — le garde-fou les écarte en bornant l'extraction et en
exigeant un lien « source » vers `src/pyrucast/`. Et deux chemins peuvent
désigner le même item : rustdoc nomme le doctest d'un `impl` paramétré
`CellGeom<'a>::det_j_w`, et fait passer un **réexport** par son chemin d'origine
— d'où une normalisation des paramètres et une comparaison par sous-suite de
segments. Chacune de ces corrections a été vérifiée en cassant volontairement le
garde-fou.

**L'audit de la prose** ne lit que les passages en `code inline`, hors blocs :
c'est ce qui écarte les noms de fichiers et les domaines, qui ressemblent à des
chemins. Il vérifie chaque segment, y compris le dernier — le découper en paires
laisserait `ops::matrix::stiffness` sans contrôle sur le nom qui bouge le plus
souvent. Et pour un `Type::membre`, il lit **la rustdoc**, qui seule connaît
l'appartenance réelle : un nom qui existe ailleurs dans le crate ne suffit pas à
valider la citation. C'est ce qui a fait tomber quatre erreurs d'un coup, dont
un `Coords::acquire` qui n'a jamais existé — en Rust le verbe est sur `Node`,
c'est côté Python qu'il est sur le magasin.

## Ce qui n'est pas vérifié, et pourquoi

Deux zones sont hors d'atteinte, et il vaut mieux les nommer que laisser croire
à une couverture totale.

**Ce qui n'est pas du code.** Une signature annotée de `/* … */`, une
énumération abrégée par `// … une ligne par physique`, un pseudo-code `f(...)` :
ces blocs se balisent ` ```text `. La coloration syntaxique mentirait, et le
lecteur voit ainsi tout de suite qu'il ne peut pas les copier. Ce n'est pas une
dérogation à la règle d'or — c'est reconnaître que ces blocs ne relèvent pas
d'elle.

La nuance se juge **bloc par bloc, jamais page par page**. Une première version
de ces conventions déclarait trois « pages d'esquisses » entières ; à les
regarder de près, elles contenaient aussi la déclaration de `Handle<T>`, celle
du trait `Cancel` et l'implémentation `PySignals` — du code réel, recopié d'un
source qui existe, et qui pourrissait comme le reste. Ces blocs sont désormais
inclus depuis `src/`, et la notion de page d'esquisses a disparu.

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

Et une règle de forme : **toute dérogation porte une raison**. Dette de
migration, item sans exemple, exclusion d'un garde-fou — chacune vit dans un
dictionnaire
nom → raison, accompagné d'un test d'hygiène qui échoue si l'entrée devient
périmée. C'est le motif déjà en place dans `test_method_exposure.py` et
`test_mirror_completeness.py` ; il n'en est pas créé d'autre.

## Où en est la mise au propre

Les deux chantiers sont terminés.

La migration du book : **257 blocs, aucun écrit à la main** — 188 Python, 69
Rust. Le registre `DETTE_MIGRATION` de `script/doc_lint.py` est vide, et le
garde-fou `fences` empêche qu'il se repeuple.

Les doctests : **`script/doc_coverage.txt` est vide**. Tout item public porte un
exemple exécutable, et le cliquet refuse désormais qu'un item nouveau entre sans
le sien. `cargo test --doc` en exécute un peu plus de neuf cents.

Une partie de la dette apparente n'en était pas une, et c'est le fait le plus
utile de l'exercice : sur les ~1530 items du départ, un tiers environ n'aurait
jamais dû figurer dans l'API publique — machinerie de mailleur `pub` par
défaut plutôt que par intention, méthodes de bibliothèques tierces entrées par
impl générique ou par `pub use rayon::prelude::*`. Écrire les exemples a servi
d'audit de la surface publique autant que de documentation.

Et écrire ces exemples a **corrigé la documentation** en une trentaine
d'endroits, chaque fois dans le même sens : l'assertion disait ce que le code
fait, la prose disait autre chose. Quelques-uns valent d'être retenus — une
symétrie réinverse la connectivité de ses mailles, `mask` rend un indicateur 0/1
et non un champ filtré, `to_poi1` dédoublonne par zone et non entre zones, et
Drucker-Prager, au sommet de son cône, laisse `p` à zéro pendant que `ε_p`
gonfle en volume.
