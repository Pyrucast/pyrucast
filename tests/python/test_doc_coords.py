"""Source des exemples Python de `book/src/coords.md`.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import pyrucast


def test_axisymetrique():
    # ANCHOR: axisymetrique
    import pyrucast

    c = pyrucast.Coords.axisymmetric()
    assert c.dim == 2 and c.is_axisymmetric
    c.add_node([1.0, 0.0])  # r = 1, z = 0
    try:
        c.add_node([-1.0, 0.0])  # x est un rayon : il doit être ≥ 0
    except RuntimeError as erreur:
        print(erreur)
    # ANCHOR_END: axisymetrique
    assert c.node_count() == 1


def test_configurations():
    # ANCHOR: configurations
    import pyrucast

    c = pyrucast.Coords(dim=2)
    n = c.add_node([0.0, 0.0])

    # Créer une deuxième configuration (clone de la configuration active).
    c2 = c.add_config("deformed")
    print(c.names())  # ['default', 'deformed']

    # Basculer sur la configuration déformée et modifier les coordonnées.
    c.select(c2)
    n.set_position([0.1, 0.05])

    # Les coordonnées lues dépendent de la configuration active.
    c.select(0)
    print(n.position())  # [0.0, 0.0]  — configuration de référence
    c.select(c2)
    print(n.position())  # [0.1, 0.05] — configuration déformée
    print(c.active)  # 1
    # ANCHOR_END: configurations
    assert c.names() == ["default", "deformed"]
    assert c.active == 1


def test_permutation():
    # ANCHOR: permutation
    import pyrucast

    c = pyrucast.Coords(dim=2)
    c.add_node([0.0, 0.0])
    c.add_node([1.0, 0.0])
    c.add_node([0.5, 1.0])

    # Affecter une permutation manuellement.
    c.set_permutation([2, 0, 1])
    print(c.permutation())  # [2, 0, 1]

    # Retour à l'identité (None = identité).
    c.clear_permutation()
    print(c.permutation())  # None
    # ANCHOR_END: permutation
    assert c.permutation() is None


def test_cycle_de_vie():
    # ANCHOR: cycle_de_vie
    import pyrucast

    c = pyrucast.Coords(dim=2)
    n = c.add_node([0.0, 0.0])  # n est un pyrucast.Node ; refcount = 1
    m = c.add_node([1.0, 0.0])

    print(c)  # Coords: dim=2, configs=1 (active="default"), nodes=2 ...
    n.set_position([0.5, 0.5])

    # GC ne touche pas tant qu'au moins un Node Python existe.
    assert c.gc() == 0

    # del + collect force le Drop côté Rust et libère le refcount.
    import gc as pygc

    del n
    pygc.collect()
    assert c.gc() == 1
    # ANCHOR_END: cycle_de_vie
