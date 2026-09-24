"""Exchange with a live **gmsh** session: meshes, physical groups and views,
both ways.

Nothing here imports ``gmsh`` at ``import pyrucast``: the module is imported
inside the functions. They only translate between the arrays of the gmsh API
and pyrucast's generic flat-array exchange (``pyrucast.mesh.from_arrays`` /
``pyrucast.export.to_arrays``), which does the work in Rust — gmsh's node
numbering included.
"""

import warnings

from ._pyrucast import (
    ElementField,
    Evolution,
    NodeField,
    element_type_from_gmsh,
    from_arrays,
    to_arrays,
)

# Fallback name for cells without a physical group. It must stay the file
# reader's own (`UNGROUPED`, src/ops/mesh/gmsh.rs): both paths return the same
# dictionary for the same mesh, that name included.
_UNGROUPED = "<ungrouped>"

_DIM = {
    "POI1": 0,
    "SEG2": 1,
    "SEG3": 1,
    "TRI3": 2,
    "TRI6": 2,
    "QUA4": 2,
    "QUA8": 2,
    "QUA9": 2,
    "TET4": 3,
    "TET10": 3,
    "PYRA5": 3,
    "PENTA6": 3,
    "PENTA15": 3,
    "HEX8": 3,
    "HEX20": 3,
    "HEX27": 3,
}


def _import_gmsh():
    """The ``gmsh`` module, or an error saying what to do."""
    try:
        import gmsh
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the gmsh exchange needs the gmsh module: pip install gmsh"
        ) from e
    if not gmsh.isInitialized():
        # Without this guard nothing would show: gmsh writes "Gmsh has not
        # been initialized" on its error output and returns empty arrays,
        # without raising. We would return an empty dict without saying why.
        raise RuntimeError(
            "gmsh is not initialized: call gmsh.initialize() and mesh "
            "before fetching the mesh"
        )
    return gmsh


def _seq(array):
    """A pyrucast ``Array`` as gmsh's API takes it best: a numpy view when numpy
    is there (no copy), a list otherwise."""
    try:
        import numpy
    except ImportError:  # pragma: no cover - depends on the environment
        return array.tolist()
    return numpy.asarray(array)


def _group_names(gmsh, dim):
    """``(dim, entity) -> [physical group names]``.

    An entity may carry several groups; the names are accumulated before
    emitting anything, so that a block's nodes are resolved only once. A group
    without a name takes ``physical <tag>``, like the file reader.
    """
    names = {}
    for gdim, gtag in gmsh.model.getPhysicalGroups(dim):
        name = gmsh.model.getPhysicalName(gdim, gtag) or f"physical {gtag}"
        for entity in gmsh.model.getEntitiesForPhysicalGroup(gdim, gtag):
            names.setdefault((gdim, int(entity)), []).append(name)
    return names


def _components(name, n):
    return [name] if n == 1 else [f"{name}_{k}" for k in range(n)]


def _views(gmsh):
    """``[(name, [(time, kind, (components, tags, values))])]`` for every
    model-based view."""
    out = []
    for tag in gmsh.view.getTags():
        index = gmsh.view.getIndex(tag)
        name = gmsh.option.getString(f"View[{index}].Name") or f"view {tag}"
        steps = int(gmsh.option.getNumber(f"View[{index}].NbTimeStep"))
        frames = []
        for step in range(steps):
            try:
                kind, tags, data, time, ncomp = gmsh.view.getHomogeneousModelData(
                    tag, step
                )
            except Exception:
                warnings.warn(
                    f"from_gmsh: view {name!r} holds no model data: skipped",
                    stacklevel=3,
                )
                frames = []
                break
            if kind not in ("NodeData", "ElementData"):
                warnings.warn(
                    f"from_gmsh: view {name!r} holds {kind}, which pyrucast "
                    "does not read: skipped",
                    stacklevel=3,
                )
                frames = []
                break
            frames.append((time, kind, (_components(name, int(ncomp)), tags, data)))
        if frames:
            out.append((name, frames))
    return out


def from_gmsh(coords, *, dim=-1, tag=-1, views=True):
    """Fetch the current gmsh model's mesh — one ``Mesh`` per named group — and
    its views.

    To be called once meshing is done on the gmsh side — pyrucast reads, it
    does not drive: geometry and meshing remain gmsh's business.

    Returns ``(meshes, fields)``:

    - ``meshes`` — the same ``dict[str, Mesh]`` as :func:`read_gmsh`, under the
      same rules: nodes land in the supplied ``coords``, whose dimension decides
      how many of gmsh's three coordinates are kept (a 2-D ``Coords`` flattens
      onto ``xy``); every returned mesh shares that ``Coords``, so a node
      between two groups is *the same* on both sides; one zone per element
      type in each group; cells without a physical group under the key
      ``"<ungrouped>"``. Named points arrive as POI1 meshes, ready to carry a
      boundary condition.
    - ``fields`` — with ``views=True``, one entry per model-based gmsh view:
      ``NodeData`` becomes a ``NodeField``, ``ElementData`` a one-point
      ``ElementField`` (a zone on every group all of whose cells it defines),
      and a view with several time steps an ``Evolution`` over their times.
      A scalar view keeps its name as component name; a view of ``n``
      components names them ``name_0 … name_{n-1}``. Other data kinds are
      skipped with a warning.

    ``dim`` restricts the import to one dimension (``-1``, the default: all),
    and ``tag`` to that single entity of dimension ``dim``. The node table is
    read in full whatever happens — a surface cell leans on nodes classified on
    its border curves — and only the nodes actually referenced are
    materialized.

    gmsh's arrays are **views** on its own memory, and pyrucast reads them
    through the buffer protocol: nothing is copied until the mesh is built. So
    ``gmsh.finalize()`` may be called right after — pyrucast then owns its data.
    """
    gmsh = _import_gmsh()
    if tag >= 0 and dim < 0:
        raise ValueError("tag is only given with a dimension: specify dim")

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    named = _group_names(gmsh, dim)

    entities = [(dim, tag)] if tag >= 0 else gmsh.model.getEntities(dim)
    blocks = []
    for edim, etag in entities:
        key = (int(edim), int(etag))
        groups = named.get(key, [_UNGROUPED])
        types, cells, connectivity = gmsh.model.mesh.getElements(*key)
        for element_type, cell_tags, conn in zip(types, cells, connectivity):
            name = element_type_from_gmsh(int(element_type))
            blocks.append((name, conn, cell_tags, groups))

    specs = _views(gmsh) if views else []
    node_specs, cell_specs, where = [], [], []
    for name, frames in specs:
        for time, kind, spec in frames:
            if kind == "NodeData":
                where.append((name, time, "node", len(node_specs)))
                node_specs.append(spec)
            else:
                where.append((name, time, "cell", len(cell_specs)))
                cell_specs.append(spec + ("cell",))

    meshes, nodal, elem = from_arrays(
        coords,
        node_tags,
        node_coords,
        blocks,
        node_fields=node_specs,
        cell_fields=cell_specs,
        order="gmsh",
    )
    frames = {}
    for name, time, kind, k in where:
        field = nodal[k] if kind == "node" else elem[k]
        frames.setdefault(name, []).append((time, field))
    fields = {
        name: steps[0][1] if len(steps) == 1 else Evolution(steps)
        for name, steps in frames.items()
    }
    return meshes, fields


def _frames(value):
    """``[(time, field)]`` for a field (one step at time 0) or an Evolution."""
    if isinstance(value, Evolution):
        return list(zip(value.shared_abscissas(), value.frames()))
    if isinstance(value, (NodeField, ElementField)):
        return [(0.0, value)]
    raise TypeError(
        "fields must map names to NodeField, ElementField or Evolution values"
    )


def _gmsh_codes():
    """pyrucast element type → gmsh element-type code."""
    codes = {}
    for code in range(1, 40):
        try:
            codes[element_type_from_gmsh(code)] = code
        except RuntimeError:
            pass
    return codes


def to_gmsh(meshes, fields=None, *, model_name="pyrucast"):
    """Push pyrucast meshes and fields into the running gmsh session, as a new
    model named ``model_name`` — to look at them in the gmsh GUI, or to save
    them with ``gmsh.write``.

    - ``meshes`` — ``dict[str, Mesh]`` on one ``Coords``: each key becomes a
      physical group (``"<ungrouped>"`` excepted). A cell present in several
      meshes is written once, on a discrete entity carrying all its groups.
    - ``fields`` — ``dict[str, NodeField | ElementField | Evolution]``: each
      becomes a view — ``NodeData`` for a node field, ``ElementData`` (the
      Gauss mean per cell) for an element field, one time step per tabulated
      value of an ``Evolution``. gmsh draws 1, 3 or 9 components; a field of
      another count gets one view per component, ``name_component``.

    gmsh must be initialized. Returns the model name.
    """
    gmsh = _import_gmsh()
    fields = fields or {}

    node_list, elem_list, where = [], [], []
    for name, value in fields.items():
        for time, field in _frames(value):
            if isinstance(field, NodeField):
                where.append((name, time, "node", len(node_list)))
                node_list.append(field)
            else:
                where.append((name, time, "cell", len(elem_list)))
                elem_list.append(field)
    arrays = to_arrays(
        meshes,
        node_fields=node_list,
        element_fields=elem_list,
        order="gmsh",
        first_tag=1,
    )

    gmsh.model.add(model_name)
    codes = _gmsh_codes()
    dim = arrays["dim"]
    xyz = list(arrays["node_coords"])
    if dim < 3:
        n = len(arrays["node_tags"])
        padded = [0.0] * (3 * n)
        for k in range(n):
            padded[3 * k : 3 * k + dim] = xyz[dim * k : dim * k + dim]
        xyz = padded

    physical = {}
    first_entity = None
    entities = []
    for name, conn, tags, groups in arrays["blocks"]:
        edim = _DIM[name]
        entity = gmsh.model.addDiscreteEntity(edim)
        entities.append((edim, entity, name, conn, tags))
        if first_entity is None or edim > first_entity[0]:
            first_entity = (edim, entity)
        for g in groups:
            if g != _UNGROUPED:
                physical.setdefault((edim, g), []).append(entity)
    if first_entity is not None:
        gmsh.model.mesh.addNodes(
            first_entity[0], first_entity[1], _seq(arrays["node_tags"]), xyz
        )
    for edim, entity, name, conn, tags in entities:
        gmsh.model.mesh.addElementsByType(entity, codes[name], _seq(tags), _seq(conn))
    for (edim, g), tags in physical.items():
        gmsh.model.addPhysicalGroup(edim, tags, name=g)

    views = {}
    for name, time, kind, k in where:
        if kind == "node":
            comps, tags, values = arrays["node_fields"][k]
            data_kind = "NodeData"
        else:
            comps, tags, values, _ = arrays["cell_fields"][k]
            data_kind = "ElementData"
        nc = len(comps)
        values = list(values)
        if nc in (1, 3, 9):
            parts = [(name, values, nc)]
        else:
            parts = [(f"{name}_{c}", values[j::nc], 1) for j, c in enumerate(comps)]
        for view_name, data, n in parts:
            view = views.get(view_name)
            if view is None:
                view = views[view_name] = [gmsh.view.add(view_name), 0]
            gmsh.view.addHomogeneousModelData(
                view[0], view[1], model_name, data_kind, _seq(tags), data, time, n
            )
            view[1] += 1
    return model_name
