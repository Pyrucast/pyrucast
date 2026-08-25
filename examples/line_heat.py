import pyrucast as pc

c = pc.Coords(1)
a = c.add_node([0.0])
b = c.add_node([1.0])
lab = pc.mesh.line(a, b, 10)
print(lab)
# lab.plot()
xc = pc.node_field.positions(lab)
print(xc)
for idx in range(len(xc[0].support_submesh())):
    print(idx, xc[0].get(idx, 0))
fes = pc.FiniteElementSpace(lab, "LAGRANGE1", "GAUSS")
print(fes)
mod1 = pc.model.heat_conduction(fes)
mat1 = pc.element_field.material_field(mod1, [("k", 1)])
mesh_T_imp = pc.mesh.poi1_from_nodes([a])
lambda_mesh = pc.mesh.barycenter(mesh_T_imp)
pv = mod1[0].primal_vars()
dv = mod1[0].dual_vars()
mod1 = mod1 | pc.model.dirichlet(pv[0], dv[0], mesh_T_imp, lambda_mesh)
K = pc.matrix.stiffness(mod1, mat1)
pa = pc.Mesh(c, "POI1")
pa[0].add_cell([b])
F = pc.NodeField(pa, mod1[0].dual_vars()) + 1
F = F + pc.NodeField(pa, mod1[0].primal_vars())[0] + 2
temp = pc.solver.solve(K, F)
