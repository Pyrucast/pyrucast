"""Modes as Python sees them: a list of dicts, written by hand or sliced."""

import itertools

import pyrucast
import pytest


def _chain():
    """Three unit springs held at their first node, a unit mass on the others."""
    coords = pyrucast.Coords(1)
    nodes = [coords.add_node([float(i)]) for i in range(4)]
    line = pyrucast.Mesh(coords, "SEG2")
    for a, b in itertools.pairwise(nodes):
        line.unit().add_cell([a, b])
    springs = pyrucast.model.heat_conduction(pyrucast.FiniteElementSpace(line))
    held = pyrucast.model.dirichlet(
        pyrucast.mesh.poi1_from_nodes(nodes[:1]), springs, "T"
    )
    model = springs | held
    materials = pyrucast.element_field.material_field(
        model, [("k", 1.0), ("rho", 1.0), ("cp", 1.0)]
    )
    stiffness = pyrucast.matrix.stiffness(model, materials)
    capacity = pyrucast.matrix.lump(pyrucast.matrix.mass(springs, materials))
    return stiffness, capacity


def test_modes_as_dicts():
    stiffness, capacity = _chain()
    modes = pyrucast.solver.eigen(stiffness, capacity, 3)
    # A slice that keeps the full support is refused, with the way out.
    with pytest.raises(RuntimeError, match="poi1_from_nodes"):
        pyrucast.matrix.project_on_basis(stiffness, modes[1:])
    # The way out: the slice on a support of its own nodes, in any order.
    part = modes[:0:-1]
    own = pyrucast.mesh.poi1_from_nodes([m["node"] for m in part])
    part = [{**m, "support": own} for m in part]
    k_part = pyrucast.matrix.project_on_basis(stiffness, part)
    for mode in part:
        node = mode["node"]
        assert k_part.get(node, "f_alpha", node, "alpha") == pytest.approx(
            mode["eigenvalue"]
        )
    # A mode written by hand is a dict like the others: a missing key is named.
    broken = [{k: v for k, v in m.items() if k != "pairs"} for m in modes]
    with pytest.raises(ValueError, match="mode 0: missing key `pairs`"):
        pyrucast.matrix.project_on_basis(stiffness, broken)
