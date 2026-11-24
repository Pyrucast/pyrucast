"""Source of the Python examples of the modal operators.

Covers `algorithmes/modes.md`. An anchor named `module-function`
(`solver-eigen`) is the example the generated reference shows under that
function. See `book/src/developper/documentation-et-tests.md`.

**The code lives at module level, not inside test functions**: mdbook does not
strip the indentation of an included excerpt, so a block anchored inside a
function would show up shifted by four spaces. pytest therefore runs this file
at **collection** time; an example that breaks is a collection error, with a
full traceback and a non-zero exit code.
"""

import math

import pyrucast


def _console():
    """A steel cantilever of 2 m, 20 Bernoulli cells, clamped at its root."""
    coords = pyrucast.Coords(3)
    root = coords.add_node([0.0, 0.0, 0.0])
    tip = coords.add_node([2.0, 0.0, 0.0])
    line = pyrucast.mesh.line(root, tip, 20)
    beam = pyrucast.model.bernoulli(
        pyrucast.FiniteElementSpace(line, interpolation="HERMITE3")
    )
    clamp = pyrucast.mesh.poi1_from_nodes([root])
    model = beam
    for variable in beam.primal_vars():
        model = model | pyrucast.model.dirichlet(clamp, beam, variable)
    materials = pyrucast.element_field.material_field(
        model,
        [
            ("E", 2e11),
            ("A", 0.01),
            ("I_y", 6.25e-6),
            ("I_z", 1e-3),
            ("J", 1e-2),
            ("G", 2e11 / 2.6),
            ("rho", 7800.0),
        ],
    )
    stiffness = pyrucast.matrix.stiffness(model, materials)
    mass = pyrucast.matrix.mass(beam, materials)
    return stiffness, mass, tip


# ── Modes propres ───────────────────────────────────────────────────────────

K, M, tip = _console()
# ANCHOR: solver-eigen
modes = pyrucast.solver.eigen(K, M, 2)  # the two modes nearest 0
frequencies = [math.sqrt(m["eigenvalue"]) / (2 * math.pi) for m in modes]
deflection = modes[0]["shape"].value(tip, "u_z")  # a NodeField, u_x … r_z
# ANCHOR_END: solver-eigen
assert abs(frequencies[0] - 17.7036) < 1e-3
assert abs(frequencies[1] - 110.7069) < 1e-3
assert all(abs(m["generalized_mass"] - 1.0) < 1e-12 for m in modes)
