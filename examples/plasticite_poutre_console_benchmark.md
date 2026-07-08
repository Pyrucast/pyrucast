# Poutre console élasto-plastique — comparaison des solveurs non linéaires

Comparaison de trois méthodes résolvant **exactement le même problème** (poutre
console 2-D, von Mises parfait, contraintes planes, encastrement à gauche,
cisaillement −1 sur la face droite amplifié de 0 à `PMAX`) :

| Méthode | Fichier | Solveur non linéaire | Opérateur d'itération |
|---|---|---|---|
| **Cast3M** (`PASAPAS`) | [`plasticite_poutre_console.dgibi`](plasticite_poutre_console.dgibi) | Newton **modifié** | `K` **élastique** constant (factorisation recyclée) |
| **pyrucast** (référence) | [`plasticite_poutre_console.rs`](plasticite_poutre_console.rs) | Newton **modifié** | `K` **élastique** constant (factorisation en cache) |
| **pyrucast + Anderson** | [`plasticite_poutre_console_anderson.rs`](plasticite_poutre_console_anderson.rs) | Newton modifié **+ accélération d'Anderson (m=3)** | `K` élastique + extrapolation sur les 3 derniers `(u, g=K⁻¹r)` |

Les **trois méthodes utilisent le même Newton modifié** : opérateur d'itération =
matrice **élastique** constante, factorisation **recyclée** d'un pas/itération à
l'autre (par défaut `PASAPAS` ne réactualise pas la matrice tangente). Les
itérations sont donc directement comparables. Anderson accélère exactement cette
même méthode en extrapolant sur l'historique récent des itérés.

Paramètres communs : `E = 210000`, `ν = 0.3`, `σy = 250`, `L = 10`, `H = 1`,
`NSTEPS = 10`, `PMAX = 5`. Machine : Linux, exécution parallèle (rayon) pour
pyrucast, mono-thread pour Cast3M.

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
| 9 | 4.5 | −8.62339e-2 | −8.623399e-2 | −8.623399e-2 | 5.73629e-4 | 5.738202e-4 | 5.738203e-4 |
| 10 | 5.0 | −9.66879e-2 | −9.668791e-2 | −9.668791e-2 | 1.00339e-3 | 1.003283e-3 | 1.003283e-3 |

Les trois méthodes coïncident (flèche à ~5-6 chiffres ; εₚ/p au 1000ᵉ près,
écarts dus aux tolérances internes distinctes, pas à la formulation).

### Itérations non linéaires (tous les pas)

| pas | Cast3M (Newton tangent) | pyrucast (Newton modifié) | Anderson m=3 |
|----:|----:|----:|----:|
| 1 | 2 | 1 | 1 |
| 2 | 2 | 1 | 1 |
| 3 | 2 | 1 | 1 |
| 4 | 2 | 1 | 1 |
| 5 | 2 | 1 | 1 |
| 6 | 2 | 1 | 1 |
| 7 | 2 | 1 | 1 |
| 8 | 5 | 16 | 7 |
| 9 | 9 | 38 | 14 |
| 10 | 12 | 89 | 29 |

Même méthode (Newton modifié, matrice élastique recyclée) pour les trois : une
itération = une descente/remontée sur la factorisation en cache, coût unitaire
comparable. Cast3M converge en beaucoup moins d'itérations que le pyrucast
d'origine à cause de **critères de convergence plus lâches** (`PRECISION`
`PASAPAS` par défaut ≈ 1e-4/1e-5 relatif, contre `tol = 1e-6` relatif ici), pas à
cause de l'opérateur. Anderson accélère la même méthode et se rapproche du compte
d'itérations de Cast3M.

### Temps d'exécution (wall-clock, médiane)

| Méthode | temps |
|---|---:|
| Cast3M `PASAPAS` (mono-thread) | ~3.12 s |
| pyrucast référence (parallèle) | ~3.32 s |
| pyrucast + Anderson (parallèle) | ~3.06 s |

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
| 7 | 3.5 | −6.70517e-2 | −6.705165e-2 | −6.705165e-2 | 2.11730e-4 | 2.116994e-4 | 2.116994e-4 |
| 8 | 4.0 | −7.66530e-2 | −7.665305e-2 | −7.665305e-2 | 6.16890e-4 | 6.170772e-4 | 6.170774e-4 |
| 9 | 4.5 | −8.63301e-2 | −8.633016e-2 | −8.633016e-2 | 1.20478e-3 | 1.203946e-3 | 1.203946e-3 |
| 10 | 5.0 | −9.68125e-2 | −9.681222e-2 | −9.681222e-2 | 1.95421e-3 | 1.952253e-3 | 1.952253e-3 |

Note : à ce maillage plus fin la première plastification apparaît dès le pas 7
(gradient de contrainte mieux résolu près de l'encastrement).

### Itérations non linéaires (tous les pas)

| pas | Cast3M (Newton tangent) | pyrucast (Newton modifié) | Anderson m=3 |
|----:|----:|----:|----:|
| 1 | 2 | 1 | 1 |
| 2 | 2 | 1 | 1 |
| 3 | 2 | 1 | 1 |
| 4 | 2 | 1 | 1 |
| 5 | 2 | 1 | 1 |
| 6 | 2 | 1 | 1 |
| 7 | 5 | 13 | 7 |
| 8 | 7 | 29 | 10 |
| 9 | 11 | 77 | 26 |
| 10 | 15 | 137 | 39 |

Même remarque qu'en 200×40 : les trois méthodes partagent le Newton modifié
(matrice élastique recyclée), itérations à coût unitaire comparable ; l'écart de
compte Cast3M ↔ pyrucast vient des critères de convergence, pas de l'opérateur.

### Temps d'exécution (wall-clock, médiane)

| Méthode | temps |
|---|---:|
| Cast3M `PASAPAS` (mono-thread) | ~16.2 s |
| pyrucast référence (parallèle) | ~22.9 s |
| pyrucast + Anderson (parallèle) | ~19.1 s |

---

## Lecture

- **Précision** : les trois méthodes donnent le même résultat aux deux tailles
  — le portage pyrucast et l'accélération d'Anderson sont validés contre Cast3M.
- **Itérations** : Anderson divise par ~3–3,5 le nombre d'itérations du Newton
  modifié sur la branche plastique (jusqu'à 137 → 39 au pas final en 400×80).
- **Temps** : l'accélération se traduit par un gain wall-clock croissant avec la
  taille — négligeable/positif en 200×40 (~3,3 → ~3,1 s), net en 400×80
  (~22,9 → ~19,1 s, ≈ −17 %). Le gain en temps reste inférieur au gain en
  itérations car chaque itération accélérée paie une évaluation de résidu
  supplémentaire (garde-fou de descente) et les pas élastiques initiaux ne
  bénéficient pas de l'accélération.
- **Même méthode** : les trois utilisent le Newton modifié avec matrice élastique
  constante et factorisation recyclée. Cast3M converge en beaucoup moins
  d'itérations (2 en élastique, 12–15 en plasticité) que le pyrucast d'origine
  (jusqu'à 137) uniquement parce que son critère de convergence est plus lâche
  (`PRECISION` `PASAPAS` par défaut ≈ 1e-4/1e-5 relatif vs. `tol = 1e-6` ici) —
  l'opérateur d'itération est identique. Anderson accélère cette même méthode et
  ramène le compte d'itérations près de celui de Cast3M, tout en gardant la
  tolérance serrée. Les temps wall-clock (Cast3M mono-thread, pyrucast parallèle)
  situent les trois sur la même machine.

_Reproduction : les binaires pyrucast lisent `PYRUCAST_NX`, `PYRUCAST_NY`,
`PYRUCAST_NSTEPS`, `PYRUCAST_PMAX` ; le script Cast3M code ces valeurs en tête de
fichier (`NX`, `NY`, `NSTEPS`, `PMAX`)._
