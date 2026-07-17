# Calcul thermique

Reprend la plaque trouée de la page [Maillage](maillage.md) pour un calcul
de **conduction thermique stationnaire**, avec les mêmes quatre types de
sollicitation que le TD Cast3M :

| Chargement | Cast3M | pyrucast |
|---|---|---|
| température imposée (bord du trou) | `BLOQ 'T'` + `DEPI` | `Model.dirichlet("T", "q", ...)` |
| flux imposé (bord gauche) | `FLUX` | `pc.assemble.flux(fes, densité, "q")` |
| convection (bord bas) | `MODE ... 'CONVECTION'` + `CONV` | `Model.convection(fes)` + `pc.assemble.flux(fes, h·T_ext, "q")` |
| source volumique (bande centrale) | `SOUR` | `pc.assemble.flux(fes, densité, "q")` sur des éléments sélectionnés |

Le point notable : en pyrucast, **`flux` est l'unique opérateur de charge
répartie**, quelle que soit la dimension du sous-espace éléments finis sur
lequel on l'applique — un bord (`SEG2` en 2D) intègre une densité linéique,
une surface intègre une densité surfacique. Il joue à la fois le rôle de
`FLUX`, `PRES` et `SOUR` en Cast3M.

## Géométrie et bords de charge

```python
{{#include ../../../formation/thermique.py:construction}}
```

> **Piège pyrucast, sans équivalent Cast3M.** En Cast3M, deux bords adjacents
> chargés (ici : le flux à gauche et la convection en bas, qui partagent un
> coin) voient leurs contributions nodales **sommées automatiquement** lors
> de l'assemblage. pyrucast combine les seconds membres par **union de
> champs** (`|`), qui exige des supports **disjoints** — deux valeurs
> différentes pour le même `(nœud, composante)` lèvent une erreur explicite
> plutôt que de se sommer silencieusement. D'où, ci-dessus, le bord bas
> **filtré** (`pyrucast.field.select` + `pyrucast.mesher.elements_on`, la
> même sélection que pour la bande source) pour exclure le coin partagé
> avec le bord gauche : c'est un choix de conception délibéré (mieux vaut
> une erreur qu'une somme implicite qui masquerait un bug), pas une
> limitation qu'on pourrait lever — mais il impose de dessiner des régions
> de charge disjointes.

## Modèle : conduction + convection + température imposée

```python
{{#include ../../../formation/thermique.py:modele}}
```

## Chargements

```python
{{#include ../../../formation/thermique.py:chargements}}
```

La sélection de la bande centrale (source volumique) suit le même principe
que le `ELEM 'APPUYE' 'STRICTEMENT'` de Cast3M : on sélectionne d'abord les
**nœuds** dans la plage voulue (`pyrucast.field.select`), puis les
**éléments qu'ils supportent intégralement**
(`pyrucast.mesher.elements_on(..., strict=True)`) — filtrer aussi en `y`
évite, ici encore, tout recouvrement avec le bord bas convecté.

## Résolution

```python
{{#include ../../../formation/thermique.py:resolution}}
```

Comme Cast3M `RESO`, `pyrucast.solver.solve` factorise la matrice creuse
(LU parallèle) et met la factorisation en cache.

![Champ de température (°C)](img/thermique.svg)

> **Non disponible dans pyrucast.**
>
> - **Rayonnement.** Pas de condition de bord de type
>   `ϕ·n = εσ(T⁴∞ − T⁴)` (Cast3M `MODE 'RAYONNEMENT'`) — seules conduction et
>   convection (film/Robin) existent.
> - **Régime transitoire.** La matrice de capacité `C = ∫ρcₚNᵢNⱼ` est
>   assemblable (`pyrucast.assemble.mass`), mais **rien ne la relie encore à
>   une boucle en temps** : chaque pas résout `[K]{T} = {P}` **stationnaire**
>   (pas de `[C]{Ṫ} + [K]{T} = {P}` intégré en temps, pas d'équivalent
>   Cast3M `PASAPAS` pour la thermique). Un `Evolution` peut faire varier
>   un chargement stationnaire d'un pas à l'autre (voir
>   [Calcul mécanique](mecanique.md) pour ce mécanisme appliqué à la
>   mécanique), mais c'est une suite de problèmes stationnaires
>   indépendants, pas une intégration temporelle au sens de Cast3M.

## Script complet

```python
{{#include ../../../formation/thermique.py}}
```

Suite : [Calcul mécanique](mecanique.md), qui réutilise ce champ de
température.
