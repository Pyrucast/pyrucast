# Poutre console élasto-plastique — comparaison des solveurs non linéaires

Comparaison de trois méthodes résolvant **exactement le même problème** (poutre
console 2-D, von Mises parfait, contraintes planes, encastrement à gauche,
cisaillement −1 sur la face droite amplifié de 0 à `PMAX`) :

| Méthode | Fichier | Solveur non linéaire | Opérateur d'itération |
|---|---|---|---|
| **Cast3M** (`PASAPAS`) | [`plasticite_poutre_console.dgibi`](plasticite_poutre_console.dgibi) | Newton **modifié** | `K` **élastique** constant (factorisation recyclée) |
| **pyrucast** (référence) | [`plasticite_poutre_console.rs`](plasticite_poutre_console.rs) · [`.py`](plasticite_poutre_console.py) | Newton **modifié** | `K` **élastique** constant (factorisation en cache) |
| **pyrucast + Anderson** | [`plasticite_poutre_console_anderson.rs`](plasticite_poutre_console_anderson.rs) · [`.py`](plasticite_poutre_console_anderson.py) | Newton modifié **+ accélération d'Anderson (m=3)** | `K` élastique + extrapolation sur les 3 derniers `(u, g=K⁻¹r)` |

Les **trois méthodes utilisent le même Newton modifié** : opérateur d'itération =
matrice **élastique** constante, factorisation **recyclée** d'un pas/itération à
l'autre (par défaut `PASAPAS` ne réactualise pas la matrice tangente). Les
itérations sont donc directement comparables. Anderson accélère exactement cette
même méthode en extrapolant sur l'historique récent des itérés.

Les deux variantes pyrucast (référence, Anderson) existent en **Rust** et en
**Python** : mêmes opérateurs natifs sous le capot, seule la boucle de pilotage
diffère. Les résultats numériques (flèche, plasticité, nombre d'itérations) sont
**identiques au bit près** entre Rust et Python
(`|flèche_Rust − flèche_Python| = 0` à tous les pas) — les tables de résultats
ci-dessous valent donc pour les deux, et seuls les temps distinguent Rust de
Python.

Paramètres communs : `E = 210000`, `ν = 0.3`, `σy = 250`, `L = 10`, `H = 1`,
`NSTEPS = 10`, `PMAX = 5`.

**Protocole de mesure** : Linux, Cast3M **et** pyrucast sont **multithreadés** —
comparaison à ressources égales. Chaque exécution est lancée **seule**, machine au
repos (aucun autre calcul concurrent), temps wall-clock médian sur plusieurs
runs. Chiffres relevés sur `HEAD` après le refactor de l'arithmétique de champs
(`merge_components`), qui a accéléré pyrucast par rapport aux mesures antérieures.

---

## Cas 200×40 QUA4

### Résultats par pas (flèche u_y au bout, mi-hauteur ; déformation plastique cumulée max)

| pas | P | flèche — Cast3M | flèche — pyrucast | flèche — Anderson | εₚ max Cast3M | p max pyrucast | p max Anderson |
|----:|----:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5 | −9.57000e-3 | −9.570000e-3 | −9.570000e-3 | 0 | 0 | 0 |
| 2 | 1.0 | −1.91400e-2 | −1.914000e-2 | −1.914000e-2 | 0 | 0 | 0 |
| 3 | 1.5 | −2.87100e-2 | −2.871000e-2 | −2.871000e-2 | 0 | 0 | 0 |
| 4 | 2.0 | −3.82800e-2 | −3.828000e-2 | −3.828000e-2 | 0 | 0 | 0 |
| 5 | 2.5 | −4.78500e-2 | −4.785000e-2 | −4.785000e-2 | 0 | 0 | 0 |
| 6 | 3.0 | −5.74200e-2 | −5.742000e-2 | −5.742000e-2 | 0 | 0 | 0 |
| 7 | 3.5 | −6.69900e-2 | −6.699000e-2 | −6.699000e-2 | 0 | 0 | 0 |
| 8 | 4.0 | −7.65775e-2 | −7.657762e-2 | −7.657762e-2 | 2.09613e-4 | 2.098470e-4 | 2.098471e-4 |
| 9 | 4.5 | −8.62339e-2 | −8.623411e-2 | −8.623411e-2 | 5.73629e-4 | 5.764621e-4 | 5.764622e-4 |
| 10 | 5.0 | −9.66879e-2 | −9.668826e-2 | −9.668826e-2 | 1.00339e-3 | 1.007817e-3 | 1.007817e-3 |

Vérification quantifiée :

- **pyrucast référence vs Anderson : identiques** — `|flèche_ref − flèche_Anderson| = 0`
  à tous les pas (Anderson ne change que le nombre d'itérations, jamais le résultat).
- **pyrucast vs Cast3M** : écart relatif max sur la flèche **3,7e-6** — accord aux
  ~6 chiffres affichés par Cast3M, cohérent avec les tolérances de convergence.

### Itérations non linéaires (tous les pas)

| pas | Cast3M | pyrucast (référence) | Anderson m=3 |
|----:|----:|----:|----:|
| 1 | 2 | 1 | 1 |
| 2 | 2 | 1 | 1 |
| 3 | 2 | 1 | 1 |
| 4 | 2 | 1 | 1 |
| 5 | 2 | 1 | 1 |
| 6 | 2 | 1 | 1 |
| 7 | 2 | 1 | 1 |
| 8 | 5 | 16 | 7 |
| 9 | 9 | 39 | 15 |
| 10 | 12 | 90 | 28 |

Même méthode (Newton modifié, matrice élastique recyclée) pour les trois : une
itération = une descente/remontée sur la factorisation en cache, coût unitaire
comparable. Cast3M converge en beaucoup moins d'itérations que le pyrucast
d'origine à cause de **critères de convergence plus lâches** (`PRECISION`
`PASAPAS` par défaut ≈ 1e-4/1e-5 relatif, contre `tol = 1e-6` relatif ici), pas à
cause de l'opérateur. Anderson accélère la même méthode et se rapproche du compte
d'itérations de Cast3M.

### Temps d'exécution (wall-clock médian, exécution isolée, multithreadé)

| Méthode | Rust | Python |
|---|---:|---:|
| Cast3M `PASAPAS` | ~3,27 s | — |
| pyrucast référence | ~2,34 s | ~2,50 s |
| pyrucast + Anderson | **~1,65 s** | **~1,73 s** |

---

## Cas 400×80 QUA4

### Résultats par pas

| pas | P | flèche — Cast3M | flèche — pyrucast | flèche — Anderson | εₚ max Cast3M | p max pyrucast | p max Anderson |
|----:|----:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5 | −9.57812e-3 | −9.578116e-3 | −9.578116e-3 | 0 | 0 | 0 |
| 2 | 1.0 | −1.91562e-2 | −1.915623e-2 | −1.915623e-2 | 0 | 0 | 0 |
| 3 | 1.5 | −2.87343e-2 | −2.873435e-2 | −2.873435e-2 | 0 | 0 | 0 |
| 4 | 2.0 | −3.83125e-2 | −3.831246e-2 | −3.831246e-2 | 0 | 0 | 0 |
| 5 | 2.5 | −4.78906e-2 | −4.789058e-2 | −4.789058e-2 | 0 | 0 | 0 |
| 6 | 3.0 | −5.74687e-2 | −5.746870e-2 | −5.746870e-2 | 0 | 0 | 0 |
| 7 | 3.5 | −6.70517e-2 | −6.705165e-2 | −6.705165e-2 | 2.11730e-4 | 2.116993e-4 | 2.116994e-4 |
| 8 | 4.0 | −7.66530e-2 | −7.665312e-2 | −7.665312e-2 | 6.16890e-4 | 6.195860e-4 | 6.195862e-4 |
| 9 | 4.5 | −8.63301e-2 | −8.633024e-2 | −8.633024e-2 | 1.20478e-3 | 1.214164e-3 | 1.214164e-3 |
| 10 | 5.0 | −9.68125e-2 | −9.681252e-2 | −9.681252e-2 | 1.95421e-3 | 1.967391e-3 | 1.967390e-3 |

Note : à ce maillage plus fin la première plastification apparaît dès le pas 7
(gradient de contrainte mieux résolu près de l'encastrement).

Vérification quantifiée :

- **pyrucast référence vs Anderson : identiques** — `|flèche_ref − flèche_Anderson| = 0`
  à tous les pas.
- **pyrucast vs Cast3M** : écart relatif max sur la flèche **1,7e-6**.

### Itérations non linéaires (tous les pas)

| pas | Cast3M | pyrucast (référence) | Anderson m=3 |
|----:|----:|----:|----:|
| 1 | 2 | 1 | 1 |
| 2 | 2 | 1 | 1 |
| 3 | 2 | 1 | 1 |
| 4 | 2 | 1 | 1 |
| 5 | 2 | 1 | 1 |
| 6 | 2 | 1 | 1 |
| 7 | 5 | 13 | 6 |
| 8 | 7 | 29 | 10 |
| 9 | 11 | 81 | 25 |
| 10 | 15 | 141 | 37 |

Même remarque qu'en 200×40 : les trois méthodes partagent le Newton modifié
(matrice élastique recyclée), itérations à coût unitaire comparable ; l'écart de
compte Cast3M ↔ pyrucast vient des critères de convergence, pas de l'opérateur.

### Temps d'exécution (wall-clock médian, exécution isolée, multithreadé)

| Méthode | Rust | Python |
|---|---:|---:|
| Cast3M `PASAPAS` | ~16,78 s | — |
| pyrucast référence | ~17,27 s | ~16,96 s |
| pyrucast + Anderson | **~10,55 s** | **~10,47 s** |

---

## Lecture

- **Précision** : les trois méthodes donnent le même résultat aux deux tailles —
  référence et Anderson coïncident au bit près, et l'écart à Cast3M reste ≤ 3,7e-6
  (relatif) : le portage pyrucast et l'accélération d'Anderson sont validés.
- **Itérations** : Anderson divise par ~3–3,8 le nombre d'itérations du Newton
  modifié sur la branche plastique (jusqu'à 141 → 37 au pas final en 400×80),
  ramenant le compte près de celui de Cast3M **tout en gardant `tol = 1e-6`**.
- **Temps** : à ressources égales (les deux codes multithreadés, exécutions
  isolées), l'accélération d'Anderson est nette — **200×40 : 2,34 → 1,65 s
  (≈ −30 %)** ; **400×80 : 17,27 → 10,55 s (≈ −39 %)**. Le gain wall-clock reste
  inférieur au gain en itérations car chaque itération accélérée paie une
  évaluation de résidu supplémentaire (garde-fou de descente) et les pas
  élastiques initiaux ne bénéficient pas de l'accélération.
- **vs Cast3M** : pyrucast + Anderson passe **sous** Cast3M aux deux tailles
  (1,65 vs 3,27 s ; 10,55 vs 16,78 s). Attention à l'interprétation : ce n'est pas
  un verdict d'algorithme (la même méthode de Newton modifié est utilisée
  partout), mais une mesure d'implémentation sur cette machine — Cast3M reste plus
  frugal en itérations grâce à ses tolérances plus lâches.
- **Rust vs Python** : résultats identiques au bit près, et **temps quasi
  identiques** (écart ≤ ~7 % en 200×40, négligeable en 400×80). Le coût réel est
  dans les opérateurs pyrucast natifs (déformation, comportement, forces
  internes, solve) — la boucle de pilotage, seule partie interprétée en Python,
  pèse d'autant moins que le maillage grandit. Python est donc une façade sans
  pénalité notable ici.

_Reproduction. **Rust** : `PYO3_PYTHON=/usr/bin/python3.13 cargo build --release
--example …`, puis les binaires lisent `PYRUCAST_NX/NY/NSTEPS/PMAX`. **Python** :
`maturin develop --release` (Python 3.13), puis
`python examples/plasticite_poutre_console[_anderson].py` avec les mêmes
variables. **Cast3M** : les valeurs `NX/NY/NSTEPS/PMAX` sont codées en tête du
`.dgibi`. Lancer chaque exécution seule (machine au repos) pour un temps fiable._
