"""Operators producing a mesh — mirror of ``ops::mesh`` (Rust).

Meshers (line, circle, arc, transfinite, paving, triangulation), sweeps and
transformations, skin and border extraction, geometric selections, gmsh
reading. Everything that returns a ``Mesh`` is here, whatever the input —
including ``select``, which extracts a field's support.

``from_gmsh`` is the only function of this module written in Python: it needs
an interpreter carrying the ``gmsh`` module, which Rust cannot have. It
merely fetches the arrays of the current gmsh model and hands them to
``from_gmsh_arrays``, which is the Rust operator proper.
"""

from ._pyrucast import (
    arc as arc,
    barycenter as barycenter,
    border as border,
    chain as chain,
    cleanup as cleanup,
    circle as circle,
    consolidate_mesh as consolidate,
    convert as convert,
    copy as copy,
    elements_on as elements_on,
    extrude as extrude,
    from_gmsh_arrays as from_gmsh_arrays,
    from_live_nodes as from_live_nodes,
    grid_surface as grid_surface,
    grid_surface2 as grid_surface2,
    invert as invert,
    line as line,
    merge_triangles as merge_triangles,
    merge_nodes as merge_nodes,
    orient as orient,
    pave_surface as pave_surface,
    pave_volume as pave_volume,
    poi1_from_nodes as poi1_from_nodes,
    points_below_plane as points_below_plane,
    points_in_cone as points_in_cone,
    points_in_cylinder as points_in_cylinder,
    points_in_sphere as points_in_sphere,
    points_in_torus as points_in_torus,
    points_on_cone as points_on_cone,
    points_on_cylinder as points_on_cylinder,
    points_on_line as points_on_line,
    points_on_plane as points_on_plane,
    points_on_sphere as points_on_sphere,
    points_on_torus as points_on_torus,
    read_gmsh as read_gmsh,
    regularize as regularize,
    read_gmsh_str as read_gmsh_str,
    revolve as revolve,
    rotate as rotate,
    select as select,
    skin as skin,
    sweep as sweep,
    sweep_solid as sweep_solid,
    symmetry_line as symmetry_line,
    symmetry_plane as symmetry_plane,
    symmetry_point as symmetry_point,
    to_poi1 as to_poi1,
    to_quadratic as to_quadratic,
    transfinite as transfinite,
    translate as translate,
    triangulate_surface as triangulate_surface,
    triangulate_volume as triangulate_volume,
)

__all__ = [
    "arc",
    "barycenter",
    "border",
    "chain",
    "cleanup",
    "circle",
    "consolidate",
    "convert",
    "elements_on",
    "extrude",
    "from_gmsh",
    "from_gmsh_arrays",
    "from_live_nodes",
    "grid_surface",
    "grid_surface2",
    "invert",
    "line",
    "merge_triangles",
    "merge_nodes",
    "orient",
    "pave_surface",
    "pave_volume",
    "poi1_from_nodes",
    "points_below_plane",
    "points_in_cone",
    "points_in_cylinder",
    "points_in_sphere",
    "points_in_torus",
    "points_on_cone",
    "points_on_cylinder",
    "points_on_line",
    "points_on_plane",
    "points_on_sphere",
    "points_on_torus",
    "read_gmsh",
    "regularize",
    "read_gmsh_str",
    "revolve",
    "rotate",
    "select",
    "skin",
    "sweep",
    "sweep_solid",
    "symmetry_line",
    "symmetry_plane",
    "symmetry_point",
    "to_poi1",
    "to_quadratic",
    "transfinite",
    "translate",
    "triangulate_surface",
    "triangulate_volume",
]


# ── Fetching the current gmsh model ─────────────────────────────────────────
# Fallback name for cells without a physical group. It must stay the file
# reader's own (`UNGROUPED`, src/ops/mesh/gmsh.rs): both paths return the same
# dictionary for the same mesh, that name included.
_UNGROUPED = "<ungrouped>"


def _import_gmsh():
    """The ``gmsh`` module, or an error saying what to do."""
    try:
        import gmsh
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "pyrucast.mesh.from_gmsh needs the gmsh module: pip install gmsh"
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


def from_gmsh(coords, *, dim=-1, tag=-1):
    """Fetches the current gmsh model's mesh, one ``Mesh`` per named group.

    To be called once meshing is done on the gmsh side — pyrucast reads, it
    does not drive: geometry and meshing remain gmsh's business.

    Returns the same ``dict[str, Mesh]`` as :func:`read_gmsh`, under the same
    rules: nodes land in the supplied ``coords``, whose dimension decides how
    many of gmsh's three coordinates are kept (a 2-D ``Coords`` flattens onto
    ``xy``); every returned mesh shares that ``Coords``, so a node between two
    groups is *the same* on both sides; one zone per element type in each
    group; cells without a physical group under the key ``"<ungrouped>"``. A
    model with no physical group at all therefore returns its whole mesh under
    that single key.

    Surfaces and points named in gmsh (``addPhysicalGroup(..., name=)``) become
    the dictionary's keys. gmsh meshes its point entities, so a named point
    arrives as a POI1 ``Mesh``, ready to carry a boundary condition.


    ``dim`` restricts the import to one dimension (``-1``, the default: all),
    and ``tag`` to that single entity of dimension ``dim``. The node table is
    read in full whatever happens — a surface cell leans on nodes classified
    on its border curves — and only the nodes actually referenced are
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
        types, _, connectivity = gmsh.model.mesh.getElements(*key)
        for element_type, conn in zip(types, connectivity):
            blocks.append((int(element_type), conn, groups))

    return from_gmsh_arrays(coords, node_tags, node_coords, blocks)
