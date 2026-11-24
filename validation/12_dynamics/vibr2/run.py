"""vibr2 — radial vibration of a thick hollow sphere (NAFEMS FV42), axisymmetric.

Problem: a steel sphere of inner radius 1.8 m and outer radius 6 m vibrates
radially. A sector of 10° is meshed in axisymmetry, and linear relations keep
the motion radial: no displacement normal to the two sides of the sector, the
same radial displacement on both sides at a given radius, and on the middle of
the sector. The first five eigenfrequencies are the radial ones.

Mesh: the `.dgibi` file's — the side P2-P3 in 5 SEG3 cells, turned by 10° about
the origin in one layer of QUA8. Its points are those of the file:
P2 = (1.8, −1.8·sin 5°), not (1.8·cos 5°, −1.8·sin 5°), so the sides sit at
∓4.98° and the middle at 0.019°; the relations keep the file's directions.

Running::

    python validation/12_dynamics/vibr2/run.py
"""

import math

import pyrucast as pc

E, NU, RHO = 2e11, 0.3, 8e3
R_IN, R_OUT = 1.8, 6.0
ALPHA = math.radians(5.0)
N_CELLS = 5


def main(out: str | None = None) -> dict[str, float]:
    ca, sa = math.cos(ALPHA), math.sin(ALPHA)
    coords = pc.Coords.axisymmetric()
    p2 = coords.add_node([R_IN, -R_IN * sa])
    p3 = coords.add_node([R_OUT, -R_OUT * sa])
    sector = pc.mesh.revolve(
        pc.mesh.line(p2, p3, N_CELLS, "SEG3"), 2.0 * ALPHA, 1, [0.0, 0.0]
    )
    elasticity = pc.model.elasticity(pc.FiniteElementSpace(sector), "axisymmetric")
    u_r, u_z = elasticity.primal_vars()

    # The nodes of each ray, from the inside out: the two sides, and the
    # middle of the sector (the mid-nodes of the arcs).
    side = math.atan2(-sa, 1.0)
    nodes = {node.id: node for zone in sector for cell in zone for node in cell}

    def ray(angle: float) -> list:
        on_ray = [
            n
            for n in nodes.values()
            if abs(math.atan2(*n.position()[::-1]) - angle) < 1e-9
        ]
        return sorted(on_ray, key=lambda n: math.hypot(*n.position()))

    d23, d41, middle = ray(side), ray(side + 2.0 * ALPHA), ray(side + ALPHA)
    on = pc.mesh.poi1_from_nodes

    # BLOQ DEPL DIRECTION on each side: no displacement along its normal.
    model = elasticity
    model = model | pc.model.mpc([(on(d23), u_r, -sa), (on(d23), u_z, -ca)], elasticity)
    model = model | pc.model.mpc([(on(d41), u_r, -sa), (on(d41), u_z, ca)], elasticity)
    # The two sides, node for node at the same radius.
    model = model | pc.model.mpc(
        [
            (on(d23), u_r, ca),
            (on(d23), u_z, sa),
            (on(d41), u_r, -ca),
            (on(d41), u_z, sa),
        ],
        elasticity,
    )
    # The corners of a side and the middle of the sector at the same radius.
    corners = on(d41[0::2])
    model = model | pc.model.mpc(
        [(corners, u_r, ca), (corners, u_z, sa), (on(middle), u_r, -1.0)], elasticity
    )

    materials = pc.element_field.material_field(
        model, [("E", E), ("nu", NU), ("rho", RHO)]
    )
    stiffness = pc.matrix.stiffness(model, materials)
    mass = pc.matrix.mass(elasticity, materials)
    modes = pc.solver.eigen(stiffness, mass, 5)
    return {
        f"f{i + 1}": math.sqrt(m["eigenvalue"]) / (2.0 * math.pi)
        for i, m in enumerate(modes)
    }


if __name__ == "__main__":
    print(main("."))
