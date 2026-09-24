"""Exchange with **medcoupling** (Salome's MED library): meshes, groups and
fields, both ways.

Nothing here is imported by ``import pyrucast``'s dependencies: ``medcoupling``
(and the ``numpy`` it brings) is imported inside the functions, the first time
one of them runs. The MED semantics — families, levels, profiles, time steps,
MED's node numbering — are read and written by medcoupling; this module only
translates between its arrays and pyrucast's generic flat-array exchange
(``pyrucast.mesh.from_arrays`` / ``pyrucast.export.to_arrays``), which does all
the work in Rust.
"""

from ._pyrucast import (
    Evolution,
    ElementField,
    NodeField,
    from_arrays,
    gauss_to_external,
    to_arrays,
)

# Cells without any group, and a MED field component without a name, get these.
_UNGROUPED = "<ungrouped>"

# pyrucast element type ↔ medcoupling geometric type (``NORM_*`` constant name).
_NORM = {
    "POI1": "NORM_POINT1",
    "SEG2": "NORM_SEG2",
    "SEG3": "NORM_SEG3",
    "TRI3": "NORM_TRI3",
    "TRI6": "NORM_TRI6",
    "QUA4": "NORM_QUAD4",
    "QUA8": "NORM_QUAD8",
    "QUA9": "NORM_QUAD9",
    "TET4": "NORM_TETRA4",
    "TET10": "NORM_TETRA10",
    "PYRA5": "NORM_PYRA5",
    "PENTA6": "NORM_PENTA6",
    "PENTA15": "NORM_PENTA15",
    "HEX8": "NORM_HEXA8",
    "HEX20": "NORM_HEXA20",
    "HEX27": "NORM_HEXA27",
}

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


def _import_medcoupling():
    """``(medcoupling, numpy)``, or an error saying what to do."""
    try:
        import medcoupling
        import numpy
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the medcoupling exchange needs the medcoupling module: "
            "pip install medcoupling"
        ) from e
    return medcoupling, numpy


def _types_by_norm(mc):
    """medcoupling's geometric-type code → pyrucast element type."""
    return {getattr(mc, norm): name for name, norm in _NORM.items()}


# ── Reading ─────────────────────────────────────────────────────────────────


def _open(mc, source, mesh_name):
    """``(MEDFileUMesh, MEDFileFields | None)`` from a path, a ``MEDFileData``
    or a ``MEDFileUMesh``."""
    if isinstance(source, str) or hasattr(source, "__fspath__"):
        source = mc.MEDFileData(str(source))
    if isinstance(source, mc.MEDFileData):
        meshes = source.getMeshes()
        names = meshes.getMeshesNames()
        if not names:
            raise ValueError("from_medcoupling: the MED data holds no mesh")
        name = names[0] if mesh_name is None else mesh_name
        if name not in names:
            raise ValueError(
                f"from_medcoupling: no mesh {name!r} (the file holds {list(names)})"
            )
        mesh = meshes.getMeshWithName(name)
        fields = source.getFields()
        return mesh, fields
    if isinstance(source, mc.MEDFileUMesh):
        return source, None
    raise TypeError(
        "from_medcoupling: source must be a path, a MEDFileData or a MEDFileUMesh"
    )


def _level_bases(mesh):
    """First cell tag of each level, so cells of all levels share one tag range."""
    bases, next_tag = {}, 0
    for lev in mesh.getNonEmptyLevels():
        bases[lev] = next_tag
        next_tag += mesh.getSizeAtLevel(lev)
    return bases


def _mesh_blocks(mc, np, mesh, bases):
    """One block per (level, type, family), plus one POI1 block per node group."""
    names = _types_by_norm(mc)
    mesh.forceComputationOfParts()
    groups_of_family = {
        mesh.getFamilyId(f): list(mesh.getGroupsOnFamily(f)) or [_UNGROUPED]
        for f in mesh.getFamiliesNames()
    }
    blocks = []
    for lev in mesh.getNonEmptyLevels():
        fam = mesh.getFamilyFieldAtLevel(lev)
        fam = fam.toNumPyArray() if fam is not None else None
        start = 0
        for gt in mesh.getGeoTypesAtLevel(lev):
            if gt not in names:
                raise ValueError(
                    f"from_medcoupling: MED cell type {mc.MEDCouplingUMesh.GetReprOfGeometricType(gt)} "
                    "has no pyrucast counterpart"
                )
            part = mesh.getDirectUndergroundSingleGeoTypeMesh(gt)
            n = part.getNumberOfCells()
            conn = part.getNodalConnectivity().toNumPyArray()
            tags = np.arange(bases[lev] + start, bases[lev] + start + n, dtype=np.int64)
            if fam is None:
                blocks.append((names[gt], conn, tags, [_UNGROUPED]))
            else:
                npc = len(conn) // max(n, 1)
                cells = conn.reshape(n, npc)
                ids = fam[start : start + n]
                for fid in np.unique(ids):
                    mask = ids == fid
                    groups = groups_of_family.get(int(fid), [_UNGROUPED])
                    blocks.append((names[gt], cells[mask].ravel(), tags[mask], groups))
            start += n
    for group in mesh.getGroupsOnSpecifiedLev(1):
        nodes = mesh.getNodeGroupArr(group).toNumPyArray()
        blocks.append(("POI1", nodes, [], [group]))
    return blocks


def _components(array, name):
    """The component names of a medcoupling array (MED allows empty ones)."""
    info = list(array.getInfoOnComponents())
    if len(info) == 1 and not info[0]:
        return [name]
    return [c or f"{name}{k}" for k, c in enumerate(info)]


def _field_steps(mc, np, fields, mesh, bases):
    """``[(name, [(time, kind, spec)], ...)]`` — one spec per time step, in the
    shape ``from_arrays`` reads: kind ``"node"`` → ``(components, tags,
    values)``, kind ``"cell"`` → ``(components, tags, values, layout)``."""
    import warnings

    names = _types_by_norm(mc)
    out = []
    if fields is None:
        return out
    for multi in fields:
        name = multi.getName()
        steps = []
        for it, order, time in multi.getTimeSteps():
            one = multi[it, order]
            kinds = one.getTypesOfFieldAvailable()
            for kind in kinds:
                if kind == mc.ON_NODES:
                    values, pfl = one.getFieldWithProfile(kind, 0, mesh)
                    tags = pfl.toNumPyArray()
                    comps = _components(values, name)
                    steps.append(
                        (time, "node", (comps, tags, values.toNumPyArray().ravel()))
                    )
                elif kind in (mc.ON_CELLS, mc.ON_GAUSS_PT):
                    for lev in one.getNonEmptyLevels()[1]:
                        values, pfl = one.getFieldWithProfile(kind, lev, mesh)
                        tags = pfl.toNumPyArray() + bases[lev]
                        comps = _components(values, name)
                        flat = values.toNumPyArray().ravel()
                        if kind == mc.ON_CELLS:
                            layout = "cell"
                        else:
                            g = one.getFieldOnMeshAtLevel(kind, lev, mesh)
                            layout = []
                            for i in range(g.getNbOfGaussLocalization()):
                                loc = g.getGaussLocalization(i)
                                layout.append(
                                    (
                                        names[loc.getType()],
                                        list(loc.getRefCoords()),
                                        list(loc.getGaussCoords()),
                                        list(loc.getWeights()),
                                    )
                                )
                        steps.append((time, "cell", (comps, tags, flat, layout)))
                else:
                    warnings.warn(
                        f"from_medcoupling: field {name!r} is defined on "
                        f"{mc.MEDCouplingFieldDiscretization.New(kind).getRepr() if hasattr(mc, 'MEDCouplingFieldDiscretization') else kind}, "
                        "which pyrucast does not read yet: skipped",
                        stacklevel=3,
                    )
        out.append((name, steps))
    return out


def from_medcoupling(coords, source, *, mesh_name=None):
    """Read a MED mesh and its fields through **medcoupling**.

    ``source`` is a path to a ``.med`` file, a ``medcoupling.MEDFileData`` or a
    ``medcoupling.MEDFileUMesh`` (mesh only). ``mesh_name`` picks one mesh of
    the file (default: the first).

    Returns ``(meshes, fields)``:

    - ``meshes`` — ``dict[str, Mesh]``, one per MED **group** (cell groups of
      every level, and node groups as POI1 meshes), all sharing ``coords``,
      whose dimension decides how many coordinates are kept. Cells of no group
      land under ``"<ungrouped>"``, as with ``read_gmsh``;
    - ``fields`` — ``dict[str, NodeField | ElementField | Evolution]``: a field
      with several time steps becomes an ``Evolution`` over its times.
      ``ON_NODES`` gives a ``NodeField``, ``ON_CELLS`` a one-point
      ``ElementField``, ``ON_GAUSS_PT`` an ``ElementField`` on the full rule —
      if MED's points are pyrucast's, otherwise it raises. A cell field gets a
      zone on every group all of whose cells it defines. Other discretizations
      (``ON_GAUSS_NE``…) are skipped with a warning.

    MED's node numbering is realigned on pyrucast's in Rust, while the arrays
    are borrowed without a copy.
    """
    mc, np = _import_medcoupling()
    mesh, fields = _open(mc, source, mesh_name)
    mesh.forceComputationOfParts()
    xyz = mesh.getCoords().toNumPyArray().ravel()
    n = mesh.getNumberOfNodes()
    node_tags = np.arange(n, dtype=np.int64)
    bases = _level_bases(mesh)
    blocks = _mesh_blocks(mc, np, mesh, bases)

    per_field = _field_steps(mc, np, fields, mesh, bases)
    node_specs, cell_specs, where = [], [], []
    for name, steps in per_field:
        for time, kind, spec in steps:
            if kind == "node":
                where.append((name, time, "node", len(node_specs)))
                node_specs.append(spec)
            else:
                where.append((name, time, "cell", len(cell_specs)))
                cell_specs.append(spec)

    meshes, nodal, elem = from_arrays(
        coords,
        node_tags,
        xyz,
        blocks,
        node_fields=node_specs,
        cell_fields=cell_specs,
        order="med",
    )

    fields_out = {}
    frames = {}
    for name, time, kind, k in where:
        field = nodal[k] if kind == "node" else elem[k]
        frames.setdefault(name, []).append((time, field))
    for name, steps in frames.items():
        fields_out[name] = steps[0][1] if len(steps) == 1 else Evolution(steps)
    return meshes, fields_out


# ── Writing ─────────────────────────────────────────────────────────────────


def _frames(value):
    """``[(time, field)]`` for a field (one step at time 0) or an Evolution."""
    if isinstance(value, Evolution):
        return list(zip(value.shared_abscissas(), value.frames()))
    if isinstance(value, (NodeField, ElementField)):
        return [(0.0, value)]
    raise TypeError(
        "fields must map names to NodeField, ElementField or Evolution values"
    )


def to_medcoupling(meshes, fields=None, *, mesh_name="mesh", gauss=True):
    """Build a ``medcoupling.MEDFileData`` from pyrucast meshes and fields —
    write it with ``.write(path, 2)``.

    - ``meshes`` — ``dict[str, Mesh]`` on one ``Coords``: each key becomes a
      MED group (``"<ungrouped>"`` excepted). A cell present in several meshes
      is written once and belongs to each group; a POI1 mesh becomes a
      **node** group.
    - ``fields`` — ``dict[str, NodeField | ElementField | Evolution]``: a
      ``NodeField`` becomes ``ON_NODES`` (``0`` where it defines nothing), an
      ``ElementField`` ``ON_GAUSS_PT`` with its rule declared in MED's
      reference element (``gauss=False``: ``ON_CELLS``, the Gauss mean per
      cell), on the cells its zones cover — a profile when that is not a whole
      level; an ``Evolution`` gives one time step per tabulated value.

    MED's node numbering is applied in Rust; the arrays are handed over
    without a copy and rearranged with numpy, never cell by cell in Python.
    """
    mc, np = _import_medcoupling()
    fields = fields or {}

    node_list, elem_list, where = [], [], []
    for name, value in fields.items():
        for step, (time, field) in enumerate(_frames(value)):
            if isinstance(field, NodeField):
                where.append((name, step, time, "node", len(node_list)))
                node_list.append(field)
            else:
                where.append((name, step, time, "cell", len(elem_list)))
                elem_list.append(field)

    arrays = to_arrays(
        meshes,
        node_fields=node_list,
        element_fields=elem_list,
        gauss=gauss,
        order="med",
        first_tag=0,
    )
    dim = arrays["dim"]
    # medcoupling takes over the memory of the numpy arrays it is given: they
    # must own it, hence the copies (views on pyrucast's arrays would dangle).
    coords = mc.DataArrayDouble(np.array(arrays["node_coords"]).reshape(-1, dim))
    blocks = [
        (name, np.asarray(conn), np.asarray(tags), groups)
        for name, conn, tags, groups in arrays["blocks"]
    ]
    n_tags = sum(len(b[2]) for b in blocks)
    cells = [b for b in blocks if b[0] != "POI1"]
    top = max((_DIM[b[0]] for b in cells), default=0)

    mm = mc.MEDFileUMesh()
    mm.setName(mesh_name)
    mm.setCoords(coords)

    # Levels: the cells of each, by MED geometric type, and where every export
    # cell tag lands — (level, index in the level mesh).
    level_of = np.full(n_tags, 1, dtype=np.int64)
    index_of = np.full(n_tags, -1, dtype=np.int64)
    level_meshes = {}
    for lev in sorted({_DIM[b[0]] - top for b in cells}, reverse=True):
        here = [b for b in cells if _DIM[b[0]] - top == lev]
        here.sort(key=lambda b: getattr(mc, _NORM[b[0]]))
        parts, count = [], 0
        for name, conn, tags, _ in here:
            part = mc.MEDCoupling1SGTUMesh(mesh_name, getattr(mc, _NORM[name]))
            part.setCoords(coords)
            part.setNodalConnectivity(mc.DataArrayInt64(np.array(conn, dtype=np.int64)))
            parts.append(part)
            level_of[tags] = lev
            index_of[tags] = np.arange(count, count + len(tags))
            count += len(tags)
        umesh = mc.MEDCoupling1SGTUMesh.AggregateOnSameCoordsToUMesh(parts)
        umesh.setName(mesh_name)
        mm.setMeshAtLevel(lev, umesh)
        level_meshes[lev] = umesh

    # Groups: cell groups per level, node groups from the POI1 blocks.
    by_level = {}
    for name, conn, tags, groups in blocks:
        for g in groups:
            if g == _UNGROUPED:
                continue
            if name == "POI1":
                by_level.setdefault(1, {}).setdefault(g, []).append(conn)
            else:
                by_level.setdefault(_DIM[name] - top, {}).setdefault(g, []).append(
                    index_of[tags]
                )
    for lev, groups in by_level.items():
        arrs = []
        for g, parts in groups.items():
            ids = mc.DataArrayInt64(np.unique(np.concatenate(parts)).astype(np.int64))
            ids.setName(g)
            arrs.append(ids)
        mm.setGroupsAtLevel(lev, arrs)

    # Fields.
    multis = {}
    for name, step, time, kind, k in where:
        one = mc.MEDFileField1TS()
        if kind == "node":
            comps, _, values = arrays["node_fields"][k]
            f = mc.MEDCouplingFieldDouble(mc.ON_NODES, mc.ONE_TIME)
            f.setName(name)
            f.setMesh(level_meshes[0])
            f.setArray(_values(mc, np, values, comps))
            f.setTime(time, step, -1)
            one.setFieldNoProfileSBT(f)
        else:
            _cell_field(
                mc,
                np,
                one,
                mm,
                level_meshes,
                level_of,
                index_of,
                blocks,
                arrays["cell_fields"][k],
                name,
                step,
                time,
                mesh_name,
            )
        multis.setdefault(name, mc.MEDFileFieldMultiTS()).pushBackTimeStep(one)

    data = mc.MEDFileData()
    ms = mc.MEDFileMeshes()
    ms.pushMesh(mm)
    data.setMeshes(ms)
    fs = mc.MEDFileFields()
    for multi in multis.values():
        fs.pushField(multi)
    data.setFields(fs)
    return data


# Corner count of each type: the nodes that fix its reference element.
_CORNERS = {
    "SEG2": 2, "SEG3": 2, "TRI3": 3, "TRI6": 3, "QUA4": 4, "QUA8": 4, "QUA9": 4,
    "TET4": 4, "TET10": 4, "PYRA5": 5, "PENTA6": 6, "PENTA15": 6,
    "HEX8": 8, "HEX20": 8, "HEX27": 8,
}  # fmt: skip


def _med_reference(mc, np, element_type, refs):
    """MED's reference element of ``element_type``, its nodes in MED order.

    The corners are medcoupling's default reference coordinates; every node is
    the affine image of pyrucast's reference element (``refs``, numbered in MED
    order by ``to_arrays``) that lands its corners there. For twelve types this
    **is** medcoupling's default element. For ``TRI6``, ``PENTA15`` and
    ``HEXA20`` medcoupling's defaults do not follow MED's connectivity (the
    first repeats a corner, the others list their middle nodes in another
    order); the element declared is then the one consistent with the
    connectivity written — which medcoupling's own Gauss-point locator does
    not accept, a limit of medcoupling rather than of the file.
    """
    norm = getattr(mc, _NORM[element_type])
    default = mc.MEDCouplingGaussLocalization.GetDefaultReferenceCoordinatesOf(norm)
    d = default.getNumberOfComponents()
    default = np.asarray(default.getValues()).reshape(-1, d)
    corners = default[: _CORNERS[element_type]]
    ours = np.asarray(refs, dtype=np.float64).reshape(-1, d)
    nc = len(corners)
    lhs = np.hstack([ours[:nc], np.ones((nc, 1))])
    fit = np.linalg.lstsq(lhs, corners, rcond=None)[0]
    placed = np.hstack([ours, np.ones((len(ours), 1))]) @ fit
    if placed.shape == default.shape and np.allclose(placed, default, atol=1e-12):
        placed = default
    return [float(x) for x in placed.ravel()]


def _values(mc, np, values, comps):
    """A ``DataArrayDouble`` of ``len(comps)`` named components."""
    arr = mc.DataArrayDouble(np.array(values, dtype=np.float64).reshape(-1, len(comps)))
    arr.setInfoOnComponents(list(comps))
    return arr


def _cell_field(
    mc,
    np,
    one,
    mm,
    level_meshes,
    level_of,
    index_of,
    blocks,
    spec,
    name,
    step,
    time,
    mesh_name,
):
    """One time step of a cell field, level by level, into ``one``."""
    comps, tags, values, layout = spec
    tags = np.asarray(tags)
    values = np.asarray(values)
    nc = len(comps)
    # Values per cell, in the order of `tags`.
    if layout == "cell":
        lengths = np.full(len(tags), nc, dtype=np.int64)
    else:
        points = {r[0]: len(r[3]) for r in layout}
        per_tag = np.zeros(len(level_of), dtype=np.int64)
        for et, _, btags, _ in blocks:
            if et in points:
                per_tag[btags] = points[et] * nc
        lengths = per_tag[tags]
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    for lev, umesh in level_meshes.items():
        sel = np.nonzero(level_of[tags] == lev)[0]
        if len(sel) == 0:
            continue
        sel = sel[np.argsort(index_of[tags[sel]], kind="stable")]
        ids = index_of[tags[sel]]
        # Gather each selected cell's run of values, vectorised.
        run = lengths[sel]
        first = np.repeat(starts[sel] - (np.cumsum(run) - run), run)
        data = values[first + np.arange(run.sum())]
        whole = len(ids) == umesh.getNumberOfCells()
        disc = mc.ON_CELLS if layout == "cell" else mc.ON_GAUSS_PT
        f = mc.MEDCouplingFieldDouble(disc, mc.ONE_TIME)
        f.setName(name)
        part = umesh if whole else umesh[mc.DataArrayInt64(ids.astype(np.int64))]
        part.setName(mesh_name)
        f.setMesh(part)
        if layout != "cell":
            for et, refs, xi, w in layout:
                norm = getattr(mc, _NORM[et])
                if norm not in part.getAllGeoTypes():
                    continue
                med = _med_reference(mc, np, et, refs)
                gx, gw = gauss_to_external(et, med, "med")
                f.setGaussLocalizationOnType(norm, med, list(gx), list(gw))
        f.setArray(_values(mc, np, data, comps))
        f.setTime(time, step, -1)
        if whole:
            one.setFieldNoProfileSBT(f)
        else:
            pfl = mc.DataArrayInt64(ids.astype(np.int64))
            pfl.setName(f"{name}_{lev}_{step}")
            one.setFieldProfile(f, mm, lev, pfl)
