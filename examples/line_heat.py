import pyrucast as pc
c = pc.Coords(1)
a = c.add_node([0.])
b = c.add_node([1.])
lab = pc.line_seg2(a,b,10)
print(lab)
#lab.plot()
xc = pc.coordinates(lab)
print(xc)
for idx in range(len(xc[0].support_submesh())):
    print(idx, xc[0].get(idx,0))
fes=pc.FiniteElementSpace(lab,"LAGRANGE1","GAUSS")
print(fes)
lab.plot(field=xc)
fes
mod1 = pc.Model.heat_conduction(fes)
mat1=pc.material_field(mod1,[('k',1)])
mesh_T_imp=pc.poi1_from_nodes([a])
lambda_mesh=pc.barycenter(mesh_T_imp)
pv=mod1[0].primal_vars()
dv=mod1[0].dual_vars()
mod1 = mod1 | pc.Model.dirichlet(pv[0],dv[0],mesh_T_imp,lambda_mesh)
K=pc.stiffness(mod1,mat1)
pa = pc.Mesh(c,'POI1')
pa[0].add_cell([b])
F=pc.NodeField(pa,mod1[0].dual_vars())+1
F=F+pc.NodeField(pa,mod1[0].primal_vars())[0]+2
temp=pc.solve(K,F)