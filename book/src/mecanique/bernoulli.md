# Poutre d'Euler-Bernoulli

## Introduction

La théorie classique des poutres : les sections planes restent planes **et
normales** à l'axe déformé, si bien que la rotation de section *est* la pente,
`θ = w'`, et qu'il n'y a aucun cisaillement transverse. C'est là toute la
différence avec [Timoshenko](timoshenko.md), le [portique 2D](portique.md) et le
[cadre 3D](cadre3d.md), qui conservent une souplesse de cisaillement.

Trois configurations partagent la même théorie :

| `model` | DDL par nœud | matériau | ce qui s'ajoute à la flexion |
|---|---|---|---|
| `planar_1d` | `w`, `theta` | `E, I` | rien — flexion pure |
| `frame_2d` | `u_x, u_y, r_z` | `+ A` | l'effort axial, et une rotation vers les axes globaux |
| `frame_3d` | 6 DDL | `+ I_y, I_z, J, G` | l'axial, la torsion, et la flexion selon **deux** axes principaux |

## Pourquoi une physique à part, et non une aire de cisaillement infinie

On pourrait atteindre Bernoulli en faisant tendre `G·A_s → ∞` dans un élément de
Timoshenko, et le résultat serait juste en arithmétique exacte. En virgule
flottante il ne l'est pas : le terme de cisaillement domine alors la raideur de
plusieurs ordres de grandeur et la réponse en flexion s'y noie — le **blocage en
cisaillement** classique, atteint par l'autre bout.

Écrire la théorie directement supprime la question, et supprime au passage deux
constantes matériau (`G`, `A_s`) qu'un modèle de Bernoulli n'a aucune raison de
demander. Réclamer une constante qu'une théorie n'utilise pas, c'est inviter la
mauvaise.

## L'élément

L'interpolation **d'Hermite cubique** du déplacement transverse rend l'élément
*exact aux nœuds* pour toute charge laissant la travée libre d'efforts répartis —
c'est pourquoi un élément par barre suffit pour un portique.

```text
           ⎡  12   6L  −12   6L ⎤
K_b = EI/L³⎢  6L  4L²  −6L  2L² ⎥        DDL [w_A, θ_A, w_B, θ_B]
           ⎢ −12  −6L   12  −6L ⎥
           ⎣  6L  2L²  −6L  4L² ⎦
```

Le repère local 3-D est déduit automatiquement d'une référence globale Z (globale
Y pour une barre quasi verticale), comme pour le [cadre 3D](cadre3d.md) : aucune
donnée d'orientation à fournir, ce qui convient aux sections symétriques.

## Variables et matériau

Le comportement (`COMP`) rend les efforts de section — `M` en 1-D, `N, M` en
plan, `N, M_y, M_z, T` dans l'espace — par une loi linéaire, comme tout élément
structural.

## Mise en donnée (Rust, testé)

```rust,ignore
{{#include ../../../tests/bernoulli.rs:example}}
```

## Exemple Python

```python
model = pyrucast.Model.bernoulli(fes, "frame_2d")
materials = pyrucast.element_field.material_field(
    model, [("E", 210_000.0), ("A", 1e-2), ("I", 1e-4)]
)
k = pyrucast.matrix.stiffness(model, materials)
```

## Compléments

**Ce que valent les tests.** Un élément de poutre gagne sa place en étant *exact
aux nœuds* : les tests comparent donc aux formules du cours à la **précision
machine** (1e-12), et non à une tolérance de discrétisation. Console sous charge
en bout (`PL³/3EI`), sous moment en bout (`ML²/2EI` — un cas qu'un signe faux
dans la matrice d'Hermite raterait tout en passant le premier), traction axiale
découplée de la flexion, et torsion `TL/GJ`.

**Quand préférer Timoshenko.** Un dernier test mesure ce qui sépare les deux
théories : une poutre **élancée** donne le même résultat aux deux (à 1 % près),
une poutre **trapue** fléchit nettement plus avec le cisaillement. Bernoulli est
exactement la théorie qui dit qu'elle ne le fait pas — à utiliser tant que
l'élancement le permet, et à quitter sinon.

> Ce test-là maille les deux poutres. Bernoulli est exact avec un seul élément,
> mais l'interpolation linéaire de Timoshenko ne l'est pas : comparer les deux
> théories sur un élément unique mesurerait le maillage, pas la physique.
