# Mécanique non linéaire

Les pages de [Mécanique linéaire](mecanique-lineaire.md) ont laissé la chape
dans un état que son acier ne supporterait pas : avec la dilatation, von Mises
y dépasse sa limite d'élasticité de 250 MPa. Les deux pages de ce chapitre
lèvent l'une puis l'autre des hypothèses qui rendaient le calcul linéaire :
- la page [Plasticité](plasticite.md) donne à l'acier une loi qui cesse d'être
  proportionnelle au-delà de sa limite ;
- la page [Contact](contact.md) remplace la masse par un appui posé dans
  l'alésage, qui ne fait que s'appuyer : le contact ne tire jamais.

Cette page pose ce que les deux ont en commun : l'équilibre, devenu non
linéaire, et la façon dont pyrucast le résout, pas à pas.

## Ce qui n'est plus linéaire

La forme discrétisée de l'équilibre reste celle de la page
[Mécanique linéaire](mecanique-lineaire.md), mais les efforts intérieurs ne
s'écrivent plus \\( [K]\\{U\\} \\) :

\\[ \\{R(U)\\} = \\{F\_{\text{ext}}\\} - \\{F\_{\text{int}}(U)\\} = 0,
\qquad \\{F\_{\text{int}}(U)\\} = \int_V [B]^T \sigma(\varepsilon(U)) \\, dV \\]

\\( \\{R\\} \\) est le **résidu** : ce qui manque à l'équilibre. Il est nul à
la solution.
- **En plasticité**, la contrainte n'est plus \\( [C]\\,\varepsilon \\) : elle
  dépend de toute l'histoire de la déformation. Une fois qu'un point a
  plastifié, il garde une déformation permanente.
- **Au contact**, ce sont les conditions aux limites qui changent : une partie
  de l'alésage touche l'appui, le reste non, et la zone touchée dépend de la
  solution elle-même.

## Le temps fictif

Une loi plastique dépend du chemin suivi : appliquer toute la charge d'un coup
ne donnerait pas le même état que la monter peu à peu. Les chargements
montent donc en **pas**, le long d'un temps qui ne sert qu'à les ordonner. Ce
**temps fictif** n'a rien à voir avec une durée : aucune vitesse n'intervient.
Chaque pas part de l'état atteint au pas précédent.

## Les itérations d'équilibre

À chaque pas, pyrucast cherche le déplacement qui annule le résidu, par
corrections successives :

```text
pour chaque pas
  appliquer les chargements de l'instant visé
  tant que l'équilibre n'est pas atteint
    résoudre  [K] {δU} = {R}        avec la rigidité élastique
    {U} ← {U} + {δU}
    intégrer la loi en chaque point de Gauss : σ, variables internes
    {R} ← {F_ext} − ∫ [B]ᵀ σ dV
  fin
fin
```

La rigidité \\( [K] \\) reste celle du matériau élastique : elle est
factorisée une fois, et chaque correction ne coûte qu'une résolution. Elle ne
suit pas la loi plastique, si bien qu'il faut plusieurs corrections là où la
matière plastifie. pyrucast les accélère en combinant les dernières
corrections entre elles.

Les blocages passent par leurs multiplicateurs, comme en linéaire ; c'est le
système de la page [Mécanique linéaire](mecanique-lineaire.md), avec le
résidu au second membre :

\\[ \begin{bmatrix} K & A^T \\\\ A & 0 \end{bmatrix}
   \begin{Bmatrix} \delta U \\\\ \lambda \end{Bmatrix} =
   \begin{Bmatrix} R \\\\ U\_{\text{imp}} - A\\,U \end{Bmatrix} \\]

**Un équilibre itéré s'arrête sur une précision.** Le pas est atteint quand le
résidu, rapporté aux forces appliquées, passe sous `precision`, et que la
déformation ne bouge presque plus d'une correction à la suivante, sous
`strain_precision`. Les deux valent \\( 10^{-4} \\) par défaut.

## Le contact, une inégalité

Le contact n'impose pas une égalité mais une inégalité : en chaque point de
l'appui, le jeu avec l'alésage reste positif ou nul, et l'effort de contact
ne peut que pousser. L'un des deux est nul : un point qui touche a un jeu nul,
un point décollé n'a pas d'effort. pyrucast résout cette alternative en
essayant des **statuts** : il suppose quels points touchent, résout, puis
corrige les points dont le jeu ou l'effort a le mauvais signe, jusqu'à ce que
tout soit cohérent. La page [Contact](contact.md) en donne le détail.

## Le parcours

1. [Plasticité](plasticite.md) : la chape de la page
   [Dilatation, matériau hétérogène](mecanique-heterogene.md), chauffée et
   chargée en 20 pas, avec un acier plastique parfait. La limite
   d'élasticité est d'abord constante, puis elle baisse avec la température.
2. [Contact](contact.md) : un appui 1 000 fois plus raide que la chape, posé
   dans l'alésage et chargé du poids, en 20 pas.

Suite : [Plasticité](plasticite.md).
