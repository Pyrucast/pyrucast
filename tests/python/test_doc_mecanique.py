"""Source des exemples Python des pages `book/src/mecanique/*.md`.

Les lois de comportement s'écrivent toutes de la même façon — déclarer le
modèle, poser le matériau, dériver la déformation, intégrer — et ces pages ne
montrent que ce qui les distingue. Le montage commun vit donc ici, hors des
ancres.

Voir `book/src/developper/documentation-et-tests.md`.
"""

import pyrucast


def _plaque_2d():
    """Un QUA4 unité, son espace EF, et un déplacement d'essai en traction."""
    c = pyrucast.Coords(2)
    n = [c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0])]
    mesh = pyrucast.Mesh(c, "QUA4")
    mesh.unit().add_cell(n)
    fes = pyrucast.FiniteElementSpace(mesh)
    u = pyrucast.NodeField(pyrucast.mesh.poi1_from_nodes(n), ["u_x", "u_y"])
    for noeud, x in zip(n, [0.0, 1.0, 1.0, 0.0]):
        u[0].set_value(noeud, "u_x", 1e-3 * x)
        u[0].set_value(noeud, "u_y", 0.0)
    return fes, u, n


def _cube_3d():
    """Un HEX8 unité : les lois en « solid » exigent un espace 3-D."""
    c = pyrucast.Coords(3)
    base = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    n = [c.add_node([float(x), float(y), float(z)]) for z in (0, 1) for x, y, _ in base]
    mesh = pyrucast.Mesh(c, "HEX8")
    mesh.unit().add_cell(n)
    fes = pyrucast.FiniteElementSpace(mesh)
    u = pyrucast.NodeField(pyrucast.mesh.poi1_from_nodes(n), ["u_x", "u_y", "u_z"])
    for noeud, (x, _, _) in zip(n, base + base):
        u[0].set_value(noeud, "u_x", 1e-3 * x)
        u[0].set_value(noeud, "u_y", 0.0)
        u[0].set_value(noeud, "u_z", 0.0)
    return fes, u, n


def _poutre_1d(dim=1):
    """Une poutre de deux SEG2, dans un `Coords` de la dimension demandée."""
    c = pyrucast.Coords(dim)
    zero = [0.0] * dim
    noeuds = []
    for i in range(3):
        p = list(zero)
        p[0] = float(i)
        noeuds.append(c.add_node(p))
    maillage = pyrucast.Mesh(c, "SEG2")
    for a, b in zip(noeuds, noeuds[1:]):
        maillage.unit().add_cell([a, b])
    return maillage, noeuds


# ── Plasticité parfaite (von Mises) ─────────────────────────────────────────


def test_plasticite_parfaite():
    fes, u, _ = _plaque_2d()
    prev_state = None
    # ANCHOR: plasticite_parfaite
    import pyrucast

    model = pyrucast.Model.plasticity_perfect(fes, "plane_stress")
    materials = pyrucast.element_field.material_field(
        model, [("E", 210_000.0), ("nu", 0.3), ("sigma_y", 250.0)]
    )

    # Déformation ε(B) issue du champ de déplacement courant (op géométrique).
    strain = pyrucast.element_field.deformation(u, fes)
    # Intégration A→B : `prev` = sortie du pas précédent (None au premier pas).
    state = pyrucast.element_field.integrate_behavior(
        model, strain, materials, prev=prev_state
    )
    sigma_xx = state[0].value(0, 0, "sigma_xx")
    p = state[0].value(0, 0, "p")  # déformation plastique cumulée
    # ANCHOR_END: plasticite_parfaite
    assert sigma_xx > 0.0
    assert p >= 0.0


# ── Lois d'écoulement : Drucker-Prager ──────────────────────────────────────


def test_drucker_prager():
    fes, u, _ = _cube_3d()
    # ANCHOR: drucker_prager
    model = pyrucast.Model.drucker_prager(fes, "solid")
    materials = pyrucast.element_field.material_field(
        model,
        [("E", 20_000.0), ("nu", 0.2), ("friction", 0.3), ("k", 30.0), ("psi", 0.1)],
    )
    strain = pyrucast.element_field.deformation(u, fes)
    state = pyrucast.element_field.integrate_behavior(model, strain, materials)
    k_t = pyrucast.matrix.tangent(model, materials, state)
    # ANCHOR_END: drucker_prager
    assert k_t.n_rows() == k_t.n_cols()


# ── Fluage de Norton ────────────────────────────────────────────────────────


def test_creep_norton():
    fes, u, _ = _cube_3d()
    # ANCHOR: creep_norton
    model = pyrucast.Model.creep_norton(fes, "solid")
    materials = pyrucast.element_field.material_field(
        model, [("E", 150_000.0), ("nu", 0.3), ("K", 400.0), ("n", 5.0)]
    )

    # Le pas de temps est obligatoire : sans lui la loi refuse d'intégrer.
    strain = pyrucast.element_field.deformation(u, fes)
    state = pyrucast.element_field.integrate_behavior(model, strain, materials, dt=1e-3)

    # La sortie devient le `prev` du pas suivant.
    state = pyrucast.element_field.integrate_behavior(
        model, strain, materials, prev=state, dt=1e-3
    )
    # ANCHOR_END: creep_norton
    assert len(state) == 1


# ── Endommagement de Mazars ─────────────────────────────────────────────────


def test_mazars():
    fes, u, _ = _plaque_2d()
    prev_state = None
    # ANCHOR: mazars
    import pyrucast

    model = pyrucast.Model.mazars(fes, "plane_stress")
    materials = pyrucast.element_field.material_field(
        model,
        [
            ("E", 30_000.0),
            ("nu", 0.2),
            ("eps_d0", 1e-4),
            ("A_t", 0.8),
            ("B_t", 20_000.0),
            ("A_c", 1.4),
            ("B_c", 1_900.0),
        ],
    )

    strain = pyrucast.element_field.deformation(u, fes)
    state = pyrucast.element_field.integrate_behavior(
        model, strain, materials, prev=prev_state
    )
    d = state[0].value(0, 0, "damage")  # endommagement scalaire D
    kappa = state[0].value(0, 0, "kappa")  # variable d'historique
    # ANCHOR_END: mazars
    assert 0.0 <= d <= 1.0
    assert kappa >= 0.0


# ── Endommagement traction/compression ──────────────────────────────────────


def test_damage_tc():
    fes, u, _ = _cube_3d()
    # ANCHOR: damage_tc
    model = pyrucast.Model.damage_tc(fes, "solid")
    materials = pyrucast.element_field.material_field(
        model,
        [
            ("E", 30_000.0),
            ("nu", 0.2),
            ("f_t", 3.0),
            ("f_c", 30.0),
            ("A_t", 0.9),
            ("A_c", 0.5),
        ],
    )
    strain = pyrucast.element_field.deformation(u, fes)
    state = pyrucast.element_field.integrate_behavior(model, strain, materials)
    # `state` porte d_plus, d_minus, r_plus, r_minus — et redevient le `prev` du pas suivant.
    # ANCHOR_END: damage_tc
    assert len(state) == 1


# ── Poutres ─────────────────────────────────────────────────────────────────


def test_bernoulli_1d():
    maillage, _ = _poutre_1d(1)
    # ANCHOR: bernoulli_1d
    fes = pyrucast.FiniteElementSpace(maillage, interpolation="HERMITE3")
    poutre = pyrucast.Model.bernoulli(fes)  # 1-D ⇒ flexion pure
    # ANCHOR_END: bernoulli_1d
    assert len(poutre) == 1


def test_bernoulli_portique():
    maillage, _ = _poutre_1d(2)
    fes = pyrucast.FiniteElementSpace(maillage, interpolation="HERMITE3")
    # ANCHOR: bernoulli_portique
    model = pyrucast.Model.bernoulli(fes)  # Coords 2-D ⇒ portique plan
    materials = pyrucast.element_field.material_field(
        model, [("E", 210_000.0), ("A", 1e-2), ("I", 1e-4)]
    )
    k = pyrucast.matrix.stiffness(model, materials)
    # ANCHOR_END: bernoulli_portique
    assert k.n_rows() == k.n_cols()


def test_timoshenko():
    maillage, _ = _poutre_1d(2)
    # ANCHOR: timoshenko
    fes = pyrucast.FiniteElementSpace(maillage, interpolation="MODEL_EMBEDDED")
    poutre = pyrucast.Model.timoshenko(fes)
    # ANCHOR_END: timoshenko
    assert len(poutre) == 1


# ── Coques ──────────────────────────────────────────────────────────────────


def test_coques():
    c = pyrucast.Coords(3)
    n = [
        c.add_node(p)
        for p in ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0])
    ]
    maillage = pyrucast.Mesh(c, "QUA4")
    maillage.unit().add_cell(n)
    fes = pyrucast.FiniteElementSpace(maillage)
    # ANCHOR: coques
    model = pyrucast.Model.shell(fes, "thick")  # ou "kirchhoff"
    materials = pyrucast.element_field.material_field(
        model, [("E", 210_000.0), ("nu", 0.3), ("h", 0.01)]
    )
    k = pyrucast.matrix.stiffness(model, materials)
    # ANCHOR_END: coques
    assert k.n_rows() == k.n_cols()


# ── Pression suiveuse ───────────────────────────────────────────────────────


def test_pression_suiveuse():
    c = pyrucast.Coords(2)
    n = [c.add_node(p) for p in ([0.0, 0.0], [1.0, 0.0])]
    maillage_de_bord = pyrucast.Mesh(c, "SEG2")
    maillage_de_bord.unit().add_cell(n)
    u = pyrucast.NodeField(pyrucast.mesh.poi1_from_nodes(n), ["u_x", "u_y"])
    # ANCHOR: pression_suiveuse
    bord = pyrucast.FiniteElementSpace(maillage_de_bord)
    charge = pyrucast.Model.follower_pressure(bord)
    materials = pyrucast.element_field.material_field(charge, [("p", 1.0e5)])

    # À chaque itération : la direction se recalcule depuis le déplacement courant.
    gradient = pyrucast.element_field.gradient(u, bord)
    traction = pyrucast.element_field.integrate_behavior(charge, gradient, materials)
    f = pyrucast.node_field.internal_forces(traction, charge)
    # ANCHOR_END: pression_suiveuse
    assert f.node_count() == 2
