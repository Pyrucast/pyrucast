"""Thermo-mécanique **pas-à-pas** au-dessus de la couche Python haut niveau.

Démo bout-en-bout de :func:`pyrucast.thermomechanics.step_by_step` : une plaque plane est chauffée
progressivement (histoire de température) et se dilate librement (appuis simples).
À chaque instant, ``step_by_step`` résout la thermique (stationnaire) puis la
mécanique non linéaire (Newton modifié + accélération d'Anderson), le couplage
étant faible — la température alimente la déformation thermique.

L'utilisateur ne fournit qu'un **dictionnaire** (maillage, modèle, charges,
matériaux, instants) ; il le récupère complété des résultats par pas. Pour une
mécanique élasto-plastique, il suffit de remplacer ``Model.elasticity`` par
``Model.plasticity`` : le même appel gère la boucle non linéaire.

Lancement ::

    maturin develop --release
    python examples/thermomecanique_pas_a_pas.py

Variables d'environnement : ``PYRUCAST_NX`` / ``PYRUCAST_NY`` (mailles),
``PYRUCAST_NSTEPS`` (pas de température).
"""

import os

import pyrucast as pc

# ── Paramètres (acier, géométrie, chargement thermique) ─────────────────────
E, NU, ALPHA = 210_000.0, 0.3, 1e-5
K_COND = 1.0
T_REF, T_HOT = 20.0, 220.0  # ΔT final = 200
LENGTH, HEIGHT = 4.0, 1.0
NX = int(os.environ.get("PYRUCAST_NX", 8))
NY = int(os.environ.get("PYRUCAST_NY", 3))
NSTEPS = int(os.environ.get("PYRUCAST_NSTEPS", 5))


def main():
    print(
        f"Plaque chauffée {NX}×{NY} QUA4 (L={LENGTH}, H={HEIGHT}), "
        f"E={E}, ν={NU}, α={ALPHA}"
    )
    print(f"Température : {T_REF} → {T_HOT} °C en {NSTEPS} pas (dilatation libre)\n")

    # ── Maillage : grille de QUA4 ───────────────────────────────────────────
    c = pc.Coords(2)
    hx, hy = LENGTH / NX, HEIGHT / NY

    def idx(i, j):
        return j * (NX + 1) + i

    grid = [c.add_node([i * hx, j * hy]) for j in range(NY + 1) for i in range(NX + 1)]
    mesh = pc.Mesh(c, "QUA4")
    for j in range(NY):
        for i in range(NX):
            mesh.unit().add_cell(
                [
                    grid[idx(i, j)],
                    grid[idx(i + 1, j)],
                    grid[idx(i + 1, j + 1)],
                    grid[idx(i, j + 1)],
                ]
            )
    fes = pc.FiniteElementSpace(mesh)

    left = [grid[idx(0, j)] for j in range(NY + 1)]
    right = [grid[idx(NX, j)] for j in range(NY + 1)]
    bottom = [grid[idx(i, 0)] for i in range(NX + 1)]

    def clamp(nodes, var, dual):
        imposed = pc.mesher.poi1_from_nodes(nodes)
        return pc.Model.dirichlet(var, dual, imposed, pc.mesher.barycenter(imposed))

    # ── Modèle : conduction + élasticité (contraintes planes) + Dirichlet ────
    # Thermique : température imposée sur les bords gauche/droit (un
    # multiplicateur par nœud ⇒ champ uniforme). Mécanique : appuis simples
    # (u_x=0 à gauche, u_y=0 en bas) ⇒ dilatation libre, sans contrainte.
    th_imposed = pc.mesher.poi1_from_nodes(left + right)
    th_mult = pc.mesher.translate(th_imposed, [0.0, 0.0])
    model = (
        pc.Model.heat_conduction(fes)
        | pc.Model.elasticity(fes, "plane_stress")
        | pc.Model.dirichlet("T", "q", th_imposed, th_mult)
        | clamp(left, "u_x", "f_x")
        | clamp(bottom, "u_y", "f_y")
    )
    materials = pc.build.material_field(
        model, [("k", K_COND), ("E", E), ("nu", NU), ("alpha", ALPHA)]
    )

    # ── Histoire de température : Evolution à valeur CHAMP (t ∈ [0, 1]) ──────
    cold = pc.NodeField(th_mult, ["imposed_T"])
    cold[0].add_to_component("imposed_T", T_REF)
    hot = pc.NodeField(th_mult, ["imposed_T"])
    hot[0].add_to_component("imposed_T", T_HOT)
    loads = pc.Evolution([(0.0, cold), (1.0, hot)], out_of_range="clamp")

    # ── Mise en donnée : un seul dictionnaire (fespace + maillage déduits du
    #    modèle) ───────────────────────────────────────────────────────────────
    data = {
        "times": [step / NSTEPS for step in range(NSTEPS + 1)],
        "model": model,
        "loads": loads,
        "materials": materials,
        "t_ref": T_REF,
    }

    # ── Calcul pas-à-pas ────────────────────────────────────────────────────
    pc.thermomechanics.step_by_step(data)

    # ── Résultats ───────────────────────────────────────────────────────────
    tip = grid[idx(NX, NY)]
    print(f"{'t':>6} {'T (°C)':>10} {'iter':>6} {'andrs':>6} {'u_x bout':>14}")
    for r in data["results"]:
        t_mean = r["temperature"].value(tip, "T")
        ux = r["displacement"].value(tip, "u_x")
        print(
            f"{r['time']:>6.2f} {t_mean:>10.3f} {r['mech_iters']:>6} "
            f"{r['mech_anderson']:>6} {ux:>14.6e}"
        )

    # Contrôle : dilatation libre ⇒ u_x = α·ΔT·L au bout.
    dT = T_HOT - T_REF
    expected = ALPHA * dT * LENGTH
    ux = data["results"][-1]["displacement"].value(tip, "u_x")
    print(f"\nu_x(bout) = {ux:.6e}   attendu α·ΔT·L = {expected:.6e}")
    assert abs(ux - expected) < 1e-7, "dilatation libre non retrouvée"

    # Export du dernier pas (déplacement) pour visualisation.
    import tempfile

    out = os.path.join(tempfile.gettempdir(), "thermomecanique_pas_a_pas.vtk")
    pc.export.export_vtk(mesh, out, data["results"][-1]["displacement"])
    print(f"Dernier pas exporté : {out}")


if __name__ == "__main__":
    main()
