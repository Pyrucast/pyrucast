"""Conduction thermique 2-D sur un maillage **importé de gmsh**.

Chaîne complète : *fichier gmsh → import → assemblage → résolution*.

Le maillage n'est pas construit à la main : on écrit un `.msh` (format gmsh
MSH 2.2 ASCII) d'un carré unité ``[0, 1]²`` maillé en QUA4, avec trois
**groupes physiques** nommés ::

    "plate"  — la surface (QUA4)
    "left"   — le bord gauche x = 0 (SEG2)
    "right"  — le bord droit  x = 1 (SEG2)

puis on le relit avec ``read_gmsh`` : on récupère un ``dict`` ``{groupe: Mesh}``
dont les régions nommées servent **directement** à poser les conditions aux
limites — tout l'intérêt des groupes physiques.

Problème (identique à ``thermal_square_2d.py``, mais le maillage vient d'un
fichier) :

  * bord **gauche** : source de chaleur répartie (flux de Neumann, densité Q) ;
  * bord **droit**  : température imposée T = 20 (Dirichlet) ;
  * bords haut/bas  : isolés (flux nul, condition naturelle).

Solution analytique, indépendante de y ::

    u(x) = 20 + (Q / k) * (1 - x)

> ``read_gmsh`` lit aussi le **binaire** gmsh ; il suffit de pointer
> ``read_gmsh(coords, "maillage.msh")`` sur un fichier produit avec
> ``-bin``, rien d'autre ne change.

Lancement ::

    maturin develop --features extension-module
    python examples/gmsh_thermal_square.py
"""

import tempfile
from pathlib import Path

import pyrucast

# ── Données du problème ──────────────────────────────────────────────────────
K = 1.0  # conductivité
Q = 10.0  # densité de flux injectée sur le bord gauche (longueur 1 ⇒ total Q)
T_IMPOSED = 20.0  # température imposée sur le bord droit
N = 4  # N×N éléments QUA4

# Codes d'éléments gmsh utilisés ici.
GMSH_SEG2 = 1
GMSH_QUA4 = 3
# Nœuds par type pyrucast (pour parcourir un maillage importé).
NODES_PER_CELL = {"POI1": 1, "SEG2": 2, "TRI3": 3, "QUA4": 4, "TET4": 4, "HEX8": 8}


def write_square_msh(path: Path, n: int) -> None:
    """Écrit un `.msh` gmsh MSH 2.2 ASCII : carré unité en n×n QUA4, avec les
    groupes physiques « plate » (surface), « left » et « right » (bords)."""
    h = 1.0 / n

    def tag(i: int, j: int) -> int:
        return j * (n + 1) + i + 1  # tags gmsh : 1-based

    lines = [
        "$MeshFormat",
        "2.2 0 8",
        "$EndMeshFormat",
        "$PhysicalNames",
        "3",
        '2 1 "plate"',
        '1 2 "left"',
        '1 3 "right"',
        "$EndPhysicalNames",
        "$Nodes",
        str((n + 1) * (n + 1)),
    ]
    for j in range(n + 1):
        for i in range(n + 1):
            lines.append(f"{tag(i, j)} {i * h} {j * h} 0")
    lines.append("$EndNodes")

    # Éléments : QUA4 (plate) + SEG2 des bords gauche/droit. Format MSH 2.2 :
    # `id type ntags phys geom noeuds...` (ntags=2 : groupe physique + entité).
    elems: list[str] = []

    def add(etype: int, phys: int, nodes: list[int]) -> None:
        eid = len(elems) + 1
        elems.append(f"{eid} {etype} 2 {phys} {phys} {' '.join(map(str, nodes))}")

    for j in range(n):
        for i in range(n):
            add(
                GMSH_QUA4,
                1,
                [tag(i, j), tag(i + 1, j), tag(i + 1, j + 1), tag(i, j + 1)],
            )
    for j in range(n):
        add(GMSH_SEG2, 2, [tag(0, j), tag(0, j + 1)])  # bord gauche
        add(GMSH_SEG2, 3, [tag(n, j), tag(n, j + 1)])  # bord droit

    lines += ["$Elements", str(len(elems)), *elems, "$EndElements"]
    path.write_text("\n".join(lines) + "\n")


def unique_nodes(mesh: "pyrucast.Mesh") -> list["pyrucast.Node"]:
    """Nœuds distincts (dédupliqués par identifiant) d'un maillage importé."""
    out: dict[int, "pyrucast.Node"] = {}
    for s, (etype, count) in enumerate(zip(mesh.element_types(), mesh.cell_counts())):
        for c in range(count):
            for k in range(NODES_PER_CELL[etype]):
                node = mesh.node(s, c, k)
                out[node.id] = node
    return list(out.values())


def main() -> None:
    # ── 1. Produire puis importer le maillage gmsh ───────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        msh = Path(tmp) / "square.msh"
        write_square_msh(msh, N)

        coords = pyrucast.Coords(dim=2)
        regions = pyrucast.mesh.read_gmsh(coords, str(msh))

    print("groupes lus :", sorted(regions))  # ['left', 'plate', 'right']
    plate = regions["plate"]
    left = regions["left"]
    right = regions["right"]

    # ── 2. Modèle thermique sur la surface + Dirichlet sur le bord droit ─────
    fes = pyrucast.FiniteElementSpace(plate)

    right_nodes = unique_nodes(right)
    imposed = pyrucast.mesh.poi1_from_nodes(right_nodes)
    multiplier = pyrucast.mesh.barycenter(imposed)
    mults = [multiplier.node(0, j, 0) for j in range(len(right_nodes))]

    model = pyrucast.model.heat_conduction(fes) | pyrucast.model.dirichlet(
        "T", "q", imposed, multiplier
    )

    # ── 3. Chargement : flux réparti sur le bord gauche + T imposée ──────────
    left_fes = pyrucast.FiniteElementSpace(left)
    model = model | pyrucast.model.flux(left_fes, "q", "thermal")
    materials = pyrucast.element_field.material_field(model, [("k", K), ("phi_q", Q)])
    source = pyrucast.node_field.external_forces(model, materials)

    imposed_mesh = pyrucast.Mesh(coords, "POI1")
    for m in mults:
        imposed_mesh.unit().add_cell([m])
    imposed_load = pyrucast.NodeField(imposed_mesh, ["imposed_T"])
    for m in mults:
        imposed_load[0].set_value(m, "imposed_T", T_IMPOSED)

    rhs = source | imposed_load

    # ── 4. Assemblage + résolution ───────────────────────────────────────────
    K_mat = pyrucast.matrix.stiffness(model, materials)
    solution = pyrucast.solver.solve(K_mat, rhs)

    # ── 5. Comparaison à l'analytique u(x) = 20 + (Q/k)(1 - x) ───────────────
    tol = 1e-9
    max_err = 0.0
    for node in unique_nodes(plate):
        x = node.position()[0]
        expected = T_IMPOSED + (Q / K) * (1.0 - x)
        got = solution.value(node, "T")
        max_err = max(max_err, abs(got - expected))
        assert abs(got - expected) < tol, f"x={x}: {got} != {expected}"

    print(f"\nerreur max sur la plaque = {max_err:.2e}")

    reaction = sum(solution.value(m, "lambda_T") for m in mults)
    print(f"réaction totale Σλ = {reaction:.6f}  (attendu {Q})")
    assert abs(reaction - Q) < tol

    # ── 6. Export VTK pour ParaView (géométrie + champ T aux nœuds) ──────────
    vtk_out = Path(tempfile.gettempdir()) / "pyrucast_plate.vtk"
    pyrucast.export.export_vtk(plate, str(vtk_out), field=solution)
    print(f"\nVTK écrit : {vtk_out}  (à ouvrir dans ParaView)")

    print("\nOK : maillage gmsh importé, résolu, exporté, conforme à l'analytique.")


if __name__ == "__main__":
    main()
