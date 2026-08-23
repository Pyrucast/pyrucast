"""Opérateurs produisant un maillage — miroir de ``ops::mesh`` (Rust).

Mailleurs (ligne, cercle, arc, transfini, pavage, triangulation), balayages
et transformations, extraction de peau et de bord, sélections géométriques,
lecture gmsh. Tout ce qui rend un ``Mesh`` est ici, quelle que soit l'entrée
— y compris ``select``, qui extrait le support d'un champ.

``from_gmsh`` est la seule fonction de ce module écrite en Python : elle a
besoin d'un interpréteur portant le module ``gmsh``, ce que Rust ne peut pas
avoir. Elle se contente d'aller chercher les tableaux du modèle gmsh courant
et de les passer à ``from_gmsh_arrays``, qui est, lui, l'opérateur Rust.
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


# ── Récupération du modèle gmsh courant ─────────────────────────────────────
# Nom de repli des mailles sans groupe physique. Doit rester celui du lecteur
# de fichier (`UNGROUPED`, src/ops/mesh/gmsh.rs) : les deux voies rendent le
# même dictionnaire pour le même maillage, ce nom compris.
_UNGROUPED = "<ungrouped>"

# Code gmsh de l'élément ponctuel, pour les points nommés que gmsh ne maille
# pas toujours (voir `_named_point_blocks`).
_GMSH_POINT = 15


def _import_gmsh():
    """Le module ``gmsh``, ou une erreur qui dit quoi faire."""
    try:
        import gmsh
    except ImportError as e:  # pragma: no cover - dépend de l'environnement
        raise ImportError(
            "pyrucast.mesh.from_gmsh a besoin du module gmsh : pip install gmsh"
        ) from e
    if not gmsh.isInitialized():
        # Sans cette garde on ne verrait rien : gmsh écrit « Gmsh has not been
        # initialized » sur sa sortie d'erreur et rend des tableaux vides,
        # sans lever. On rendrait un dictionnaire vide sans dire pourquoi.
        raise RuntimeError(
            "gmsh n'est pas initialisé : appelez gmsh.initialize() et maillez "
            "avant de récupérer le maillage"
        )
    return gmsh


def _group_names(gmsh, dim):
    """``(dim, entité) -> [noms de groupes physiques]``.

    Une entité peut porter plusieurs groupes ; on accumule les noms avant
    d'émettre quoi que ce soit, pour ne résoudre les nœuds d'un bloc qu'une
    fois. Un groupe sans nom prend ``physical <tag>``, comme le lecteur de
    fichier.
    """
    names = {}
    for gdim, gtag in gmsh.model.getPhysicalGroups(dim):
        name = gmsh.model.getPhysicalName(gdim, gtag) or f"physical {gtag}"
        for entity in gmsh.model.getEntitiesForPhysicalGroup(gdim, gtag):
            names.setdefault((gdim, int(entity)), []).append(name)
    return names


def _named_point_blocks(gmsh, dim):
    """Blocs POI1 pour les groupes physiques de dimension 0 restés sans maille.

    gmsh ne crée pas toujours d'élément ponctuel sur un point nommé. Le groupe
    existe pourtant, et c'est souvent celui sur lequel on veut poser une
    condition : on le récupère par ses nœuds. Un POI1 porte un nœud par maille,
    donc la liste des tags *est* la connectivité — rien à faire côté Rust.

    Réservé à la dimension 0 : ailleurs, ce repli transformerait silencieusement
    une surface en nuage de points.
    """
    if dim not in (-1, 0):
        return []
    blocks = []
    for gdim, gtag in gmsh.model.getPhysicalGroups(0):
        entities = gmsh.model.getEntitiesForPhysicalGroup(gdim, gtag)
        if any(len(gmsh.model.mesh.getElements(0, int(e))[0]) for e in entities):
            continue  # gmsh a bien maillé ces points, la voie normale suffit
        tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(gdim, gtag)
        if len(tags):
            name = gmsh.model.getPhysicalName(gdim, gtag) or f"physical {gtag}"
            blocks.append((_GMSH_POINT, tags, [name]))
    return blocks


def from_gmsh(coords, *, dim=-1, tag=-1):
    """Récupère le maillage du modèle gmsh courant, un ``Mesh`` par groupe nommé.

    À appeler une fois le maillage terminé côté gmsh — pyrucast lit, il ne
    pilote pas : la géométrie et le maillage restent l'affaire de gmsh.

    Rend le même ``dict[str, Mesh]`` que :func:`read_gmsh`, avec les mêmes
    règles : les nœuds atterrissent dans le ``coords`` fourni, dont la dimension
    décide combien des trois coordonnées de gmsh sont gardées (un ``Coords``
    2-D aplatit sur ``xy``) ; tous les maillages rendus partagent ce ``Coords``,
    donc un nœud entre deux groupes est *le même* des deux côtés ; une zone par
    type d'élément dans chaque groupe ; les mailles sans groupe physique sous
    la clé ``"<ungrouped>"``. Un modèle sans aucun groupe physique rend donc
    tout son maillage sous cette seule clé.

    Les surfaces et les points nommés dans gmsh (``addPhysicalGroup(..., name=)``)
    deviennent les clés du dictionnaire. Un point nommé que gmsh n'a pas maillé
    est quand même rendu, sous forme de nuage POI1 de ses nœuds.

    ``dim`` restreint l'import à une dimension (``-1``, le défaut : toutes), et
    ``tag`` à cette seule entité de dimension ``dim``. La table des nœuds est
    lue en entier quoi qu'il arrive — une maille de surface s'appuie sur des
    nœuds classés sur ses courbes de bord —, et seuls les nœuds effectivement
    référencés sont matérialisés.

    Les tableaux de gmsh sont des **vues** sur sa propre mémoire, et pyrucast
    les lit par le protocole tampon : rien n'est copié jusqu'à la construction
    du maillage. On peut donc appeler ``gmsh.finalize()`` juste après — pyrucast
    possède alors ses données.
    """
    gmsh = _import_gmsh()
    if tag >= 0 and dim < 0:
        raise ValueError("tag ne se donne qu'avec une dimension : précisez dim")

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
    blocks.extend(_named_point_blocks(gmsh, dim))

    return from_gmsh_arrays(coords, node_tags, node_coords, blocks)
