```mermaid
classDiagram
    class viz_Bbox3 {
        +min: Point3
        +max: Point3
        +empty() Self
        +extend(p:Point3) void
        +is_empty() bool
        +center() Point3
        +diagonal() f64
    }
    class viz_Projector {
        +right: Vector3
        +up: Vector3
        +forward: Vector3
        +target: Point3
        +new(view:&View, default_target:Point3) Self
        +project(p:Point3) Vector3
    }
    class viz_MeshFieldView {
        +mesh: &'a Mesh
        +field: &'a NodeField
        +component: &'a str
    }
    class viz_Drawable {
        +bbox() Result~viz_Bbox3~
        +draw_on(area:&DrawingArea~DB, Shift~, view:&View) ~DB: DrawingBackend~
        +bbox() Result~viz_Bbox3~
        +draw_on(area:&DrawingArea~DB, Shift~, view:&View) ~DB: DrawingBackend~
    }
    class viz_SubMeshFieldView {
        +submesh: &'a SubMesh
        +field: &'a NodeField
        +component: &'a str
    }
    class viz_ProjPrim {
        +depth() f64
    }
    class viz_Drawable {
        +bbox() Result~viz_Bbox3~
        +draw_on(area:&DrawingArea~DB, Shift~, view:&View) ~DB: DrawingBackend~
        +bbox() Result~viz_Bbox3~
        +draw_on(area:&DrawingArea~DB, Shift~, view:&View) ~DB: DrawingBackend~
    }
    class viz_View {
        +yaw: f64
        +pitch: f64
        +scale: f64
        +target: Option~crate::mesh::point::Point3~
        +show_axes: bool
        +front() Self
        +top() Self
        +side() Self
        +iso() Self
    }
    class viz_Default {
        +default() Self
    }
    class viz_SaveFormat {
        +from_path(path:&Path) Result~Self~
    }
    class viz_App {
        -object: &'a D
        -field_button: Option~&'a dyn FieldButton~
        -target: crate::mesh::point::Point3
        -yaw: f64
        -pitch: f64
        -scale: f64
        -show_axes: bool
        -width: u32
        -height: u32
        -pixel_buf: Vec~u8~
        -window: Option~Rc<Window>~
        -surface: Option~softbuffer::Surface<Rc<Window>, Rc<Window>>~
        -dragging: bool
        -last_mouse: Option~(f64, f64)~
        -cursor: Option~(f64, f64)~
    }
    class viz_ApplicationHandler {
        +resumed(event_loop:&ActiveEventLoop) void
        +window_event(event_loop:&ActiveEventLoop, _id:WindowId, event:WindowEvent) void
    }
    class viz_FieldDrawable {
        -source: FieldSource~'a~
        -field: &'a crate::containers::node_field::NodeField
        -components: Vec~String~
        -selected: mesh_Cell~usize~
    }
    class viz_Drawable {
        +bbox() Result~viz_Bbox3~
        +draw_on(area:&plotters::drawing::DrawingArea~DB, plotters::coord::Shift~, view:&View) ~DB: DrawingBackend~
    }
    class viz_FieldButton {
        +cycle() void
    }
    class py_PyCell {
        -inner: mesh_Cell
        +from_cell(c:mesh_Cell) Self
        +index() usize
        +element_type() PyResult~String~
        +node_ids() PyResult~Vec<u32>~
        +nodes() PyResult~Vec<PyNode>~
        +__len__() PyResult~usize~
        +__getitem__(idx:isize) PyResult~py_PyNode~
        +__repr__() PyResult~String~
        +__str__() PyResult~String~
    }
    class py_PyConfiguration {
        -handle: Handle~mesh_Configuration~
        +py_new(dim:u8) PyResult~Self~
        +dim() PyResult~u8~
        +node_count() PyResult~usize~
        +capacity() PyResult~usize~
        +is_alive(id:u32) PyResult~bool~
        +add_node(coords:Vec~f64~) PyResult~py_PyNode~
        +acquire(id:u32) PyResult~py_PyNode~
        +refcount(id:u32) PyResult~u32~
        +gc() PyResult~usize~
        +coord(id:u32) PyResult~Vec<f64>~
        +set_coord(id:u32, coords:Vec~f64~) PyResult~()~
        +add_coord_set(name:String) PyResult~usize~
        +switch_to(set:usize) PyResult~()~
        +active_set() PyResult~usize~
        +set_names() PyResult~Vec<String>~
        +permutation() PyResult~Option<Vec<u32>>~
        +set_permutation(perm:Vec~u32~) PyResult~()~
        +clear_permutation() PyResult~()~
        +__repr__() PyResult~String~
        +__str__() PyResult~String~
    }
    class py_PySubElementField {
        -handle: Handle~containers_SubElementField~
        +py_new(fespace:PyRef~py_PySubFiniteElementSpace~, components:Vec~String~) PyResult~Self~
        +from_uniform_per_component(_cls:&pyo3::Bound~'_, pyo3::types::PyType~, fespace:PyRef~py_PySubFiniteElementSpace~, components:Vec~String~, values_per_component:Vec~f64~) PyResult~Self~
        +cell_count() PyResult~usize~
        +gauss_count() PyResult~usize~
        +component_count() PyResult~usize~
        +components() PyResult~Vec<String>~
        +component_index(name:&str) PyResult~Option<usize>~
        +get(cell:usize, gauss:usize, comp:usize) PyResult~f64~
        +set(cell:usize, gauss:usize, comp:usize, value:f64) PyResult~()~
        +value(cell:usize, gauss:usize, component:&str) PyResult~f64~
        +set_value(cell:usize, gauss:usize, component:&str, value:f64) PyResult~()~
        +point_values(cell:usize, gauss:usize) PyResult~Vec<f64>~
        +set_uniform(component:&str, value:f64) PyResult~()~
        +set_cell_uniform(cell:usize, component:&str, value:f64) PyResult~()~
        +add_to_component(component:&str, scalar:f64) PyResult~()~
        +sub_to_component(component:&str, scalar:f64) PyResult~()~
        +mul_to_component(component:&str, scalar:f64) PyResult~()~
        +div_to_component(component:&str, scalar:f64) PyResult~()~
        +__add__(rhs:f64) PyResult~py_PySubElementField~
        +__sub__(rhs:f64) PyResult~py_PySubElementField~
        +__mul__(rhs:f64) PyResult~py_PySubElementField~
        +__truediv__(rhs:f64) PyResult~py_PySubElementField~
        +__getitem__(key:(usize, usize, String)) PyResult~f64~
        +__setitem__(key:(usize, usize, String), value:f64) PyResult~()~
        +__repr__() PyResult~String~
        +__str__() PyResult~String~
    }
    class py_PyElementField {
        -inner: containers_ElementField
        +py_new(fespace:PyRef~py_PyFiniteElementSpace~, components:Vec~String~) PyResult~Self~
        +with_components_per_subspace(_cls:&pyo3::Bound~'_, pyo3::types::PyType~, fespace:PyRef~py_PyFiniteElementSpace~, components_per_subspace:Vec~Vec<String>~) PyResult~Self~
        +subfield_count() PyResult~usize~
        +subfield(i:usize) PyResult~py_PySubElementField~
        +__repr__() PyResult~String~
        +__str__() PyResult~String~
    }
    class py_PySubFiniteElementSpace {
        -handle: Handle~finite_element_space_SubFiniteElementSpace~
        +element_type() PyResult~String~
        +interpolation() PyResult~String~
        +quadrature() PyResult~String~
        +ref_dim() PyResult~usize~
        +space_dim() PyResult~usize~
        +nodes_per_cell() PyResult~usize~
        +cell_count() PyResult~usize~
        +gauss_count() PyResult~usize~
        +gauss_xi(g:usize) PyResult~Vec<f64>~
        +gauss_weight(g:usize) PyResult~f64~
        +n_at_g(g:usize) PyResult~Vec<f64>~
        +dn_at_g(g:usize) PyResult~Vec<f64>~
        +jacobian(cell_idx:usize, g:usize) PyResult~Vec<f64>~
        +det_jacobian(cell_idx:usize, g:usize) PyResult~f64~
        +dn_dx(cell_idx:usize, g:usize) PyResult~Vec<f64>~
        +__repr__() PyResult~String~
        +__str__() PyResult~String~
    }
    class py_PyFiniteElementSpace {
        -inner: finite_element_space_FiniteElementSpace
        +py_new(mesh:PyRef~py_PyMesh~, interpolation:&str, quadrature:&str) PyResult~Self~
        +with_choices(_cls:&pyo3::Bound~'_, pyo3::types::PyType~, mesh:PyRef~py_PyMesh~, choices:Vec~(String, String)~) PyResult~Self~
        +lagrange1(_cls:&pyo3::Bound~'_, pyo3::types::PyType~, mesh:PyRef~py_PyMesh~) PyResult~Self~
        +subspace_count() PyResult~usize~
        +subspace(i:usize) PyResult~py_PySubFiniteElementSpace~
        +__repr__() PyResult~String~
        +__str__() PyResult~String~
    }
    class py_PyMatrix {
        -handle: Handle~containers_Matrix~
        +py_new(symmetric:bool) PyResult~Self~
        +add_entry(row_node:u32, row_field:&str, col_node:u32, col_field:&str, value:f64) PyResult~()~
        +get(row_node:u32, row_field:&str, col_node:u32, col_field:&str) PyResult~f64~
        +n_rows() PyResult~usize~
        +n_cols() PyResult~usize~
        +entry_count() PyResult~usize~
        +symmetric() PyResult~bool~
        +field_names() PyResult~Vec<String>~
        +row_dofs() PyResult~Vec<(u32, String)>~
        +col_dofs() PyResult~Vec<(u32, String)>~
        +dense() PyResult~Vec<f64>~
        +mul_dense(x:Vec~f64~) PyResult~Vec<f64>~
        +entries() PyResult~Vec<(u32, String, u32, String, f64)>~
        +__len__() PyResult~usize~
        +__repr__() PyResult~String~
        +__str__() PyResult~String~
    }
    class py_PySubModel {
        -handle: Handle~containers_SubModel~
        +heat_conduction(_cls:&pyo3::Bound~'_, pyo3::types::PyType~, fespace:PyRef~py_PySubFiniteElementSpace~, material:PyRef~py_PySubElementField~) PyResult~Self~
        +dirichlet(_cls:&pyo3::Bound~'_, pyo3::types::PyType~, config:PyRef~py_PyConfiguration~, primal_var:String, primal_dual:String, constrained_node_ids:Vec~u32~) PyResult~Self~
        +primal_vars() PyResult~Vec<String>~
        +dual_vars() PyResult~Vec<String>~
        +multiplier_nodes() PyResult~Vec<u32>~
        +__repr__() PyResult~String~
        +__str__() PyResult~String~
    }
    class py_PyModel {
        -inner: containers_Model
        +py_new() PyResult~Self~
        +add_sub_model(sub:PyRef~py_PySubModel~) PyResult~()~
        +sub_model_count() PyResult~usize~
        +sub_model(i:usize) PyResult~py_PySubModel~
        +primal_vars() PyResult~Vec<String>~
        +dual_vars() PyResult~Vec<String>~
        +stiffness() PyResult~py_PyMatrix~
        +mass() PyResult~py_PyMatrix~
        +__repr__() PyResult~String~
        +__str__() PyResult~String~
    }
    class py_PyNode {
        -node: mesh_Node
        +from_raw(handle:Handle~mesh_Configuration~, id:mesh_NodeId) Self
        +from_node(node:mesh_Node) Self
        +as_node() &Node
        +id() u32
        +coord() PyResult~Vec<f64>~
        +set_coord(coords:Vec~f64~) PyResult~()~
        +__repr__() PyResult~String~
        +__str__() PyResult~String~
    }
    class py_PyNodeField {
        -handle: Handle~containers_NodeField~
        +py_new(submesh:PyRef~py_PySubMesh~, components:Vec~String~) PyResult~Self~
        +node_count() PyResult~usize~
        +component_count() PyResult~usize~
        +components() PyResult~Vec<String>~
        +get(node_idx:usize, comp_idx:usize) PyResult~f64~
        +set(node_idx:usize, comp_idx:usize, value:f64) PyResult~()~
        +get_by_node(node_id:u32, comp_idx:usize) PyResult~f64~
        +set_by_node(node_id:u32, comp_idx:usize, value:f64) PyResult~()~
        +component_index(name:&str) PyResult~Option<usize>~
        +node_values(node_idx:usize) PyResult~Vec<f64>~
        +to_poi1_submesh() PyResult~py_PySubMesh~
        +to_poi1_mesh() PyResult~py_PyMesh~
        +value(node_id:u32, component:&str) PyResult~f64~
        +set_value(node_id:u32, component:&str, value:f64) PyResult~()~
        +add_fields(other:PyRef~py_PyNodeField~) PyResult~py_PyNodeField~
        +merge_fields(other:PyRef~py_PyNodeField~) PyResult~py_PyNodeField~
        +add_to_component(component:&str, scalar:f64) PyResult~()~
        +sub_to_component(component:&str, scalar:f64) PyResult~()~
        +mul_to_component(component:&str, scalar:f64) PyResult~()~
        +div_to_component(component:&str, scalar:f64) PyResult~()~
        +restrict(mesh:PyRef~py_PyMesh~) PyResult~py_PyNodeField~
        +__add__(rhs:f64) PyResult~py_PyNodeField~
        +__sub__(rhs:f64) PyResult~py_PyNodeField~
        +__mul__(rhs:f64) PyResult~py_PyNodeField~
        +__truediv__(rhs:f64) PyResult~py_PyNodeField~
        +__getitem__(key:(u32, String)) PyResult~f64~
        +__setitem__(key:(u32, String), value:f64) PyResult~()~
        +__repr__() PyResult~String~
        +__str__() PyResult~String~
    }
    class py_PySubMesh {
        -handle: Handle~mesh_SubMesh~
        +py_new(config:PyRef~py_PyConfiguration~, element_type:&str) PyResult~Self~
        +element_type() PyResult~String~
        +add_cell(nodes:Vec~u32~) PyResult~usize~
        +cell_count() PyResult~usize~
        +face_color() PyResult~(u8, u8, u8)~
        +set_face_color(rgb:(u8, u8, u8)) PyResult~()~
        +plot(view:Option~(f64, f64, f64)~, save:Option~std::path::PathBuf~, show_axes:bool, field:Option~PyRef<crate::py::node_field::PyNodeField>~, component:Option~String~) PyResult~()~
        +__len__() PyResult~usize~
        +__getitem__(idx:isize) PyResult~crate::py::cell::PyCell~
        +__repr__() PyResult~String~
        +__str__() PyResult~String~
    }
    class py_PyMesh {
        -inner: mesh_Mesh
        +py_new(config:PyRef~py_PyConfiguration~, element_type:Option~&str~) PyResult~Self~
        +add_submesh(sm:PyRef~py_PySubMesh~) PyResult~()~
        +add_cell(nodes:Vec~u32~) PyResult~usize~
        +element_type() PyResult~Option<String>~
        +element_types() PyResult~Vec<String>~
        +cell_counts() PyResult~Vec<usize>~
        +node(submesh_idx:usize, cell_idx:usize, node_idx:usize) PyResult~py_PyNode~
        +from_live_nodes(_cls:&pyo3::Bound~'_, pyo3::types::PyType~, config:PyRef~py_PyConfiguration~) PyResult~Self~
        +line_seg2(_cls:&pyo3::Bound~'_, pyo3::types::PyType~, a:PyRef~py_PyNode~, b:PyRef~py_PyNode~, n_elems:usize) PyResult~Self~
        +circle_seg2(_cls:&pyo3::Bound~'_, pyo3::types::PyType~, center:PyRef~py_PyNode~, normal:Vec~f64~, radius:f64, n_elems:usize) PyResult~Self~
        +sweep_qua4(_cls:&pyo3::Bound~'_, pyo3::types::PyType~, mesh_a:PyRef~py_PyMesh~, mesh_b:PyRef~py_PyMesh~, n_layers:usize) PyResult~Self~
        +extrude(_cls:&pyo3::Bound~'_, pyo3::types::PyType~, mesh:PyRef~py_PyMesh~, direction:Vec~f64~, n_layers:usize) PyResult~Self~
        +fill_surface(_cls:&pyo3::Bound~'_, pyo3::types::PyType~, contour:PyRef~py_PyMesh~, element_type:&str, max_edge_length:Option~f64~, min_angle_deg:Option~f64~) PyResult~Self~
        +__add__(other:PyRef~py_PyMesh~) PyResult~py_PyMesh~
        +consolidate() PyResult~py_PyMesh~
        +submesh_count() PyResult~usize~
        +submesh(idx:usize) PyResult~py_PySubMesh~
        +cell(submesh_idx:usize, cell_idx:usize) PyResult~crate::py::cell::PyCell~
        +cell_count() PyResult~usize~
        +plot(view:Option~(f64, f64, f64)~, save:Option~std::path::PathBuf~, show_axes:bool, field:Option~PyRef<crate::py::node_field::PyNodeField>~, component:Option~String~) PyResult~()~
        +__repr__() PyResult~String~
        +__str__() PyResult~String~
    }
    class containers_DofId {
        +node_id: mesh_NodeId
        +field_idx: u32
    }
    class containers_Matrix {
        -field_names: Vec~String~
        -row_dofs: Vec~containers_DofId~
        -col_dofs: Vec~containers_DofId~
        -entries: Vec~(u32, u32, f64)~
        -symmetric: bool
        +new(symmetric:bool) Self
        +symmetric() bool
        +n_rows() usize
        +n_cols() usize
        +entry_count() usize
        +field_names() &[String]
        +field_name(idx:u32) &str
        +field_index(name:&str) Option~u32~
        +row_dofs() &[DofId]
        +col_dofs() &[DofId]
        +add_entry(row_node:mesh_NodeId, row_field:&str, col_node:mesh_NodeId, col_field:&str, value:f64) void
        +get(row_node:mesh_NodeId, row_field:&str, col_node:mesh_NodeId, col_field:&str) f64
        +iter_entries() impl Iterator~Item = (DofId, containers_DofId, f64)~
        +dense() Vec~f64~
        +to_dmatrix() DMatrix~f64~
        +to_coo() CooMatrix~f64~
        +to_csr() CsrMatrix~f64~
        +to_csc() CscMatrix~f64~
        +mul_dense(x:&[f64]) Result~Vec<f64>~
        +intern_field(name:&str) u32
        +find_or_insert_row(node_id:mesh_NodeId, field_idx:u32) u32
        +find_or_insert_col(node_id:mesh_NodeId, field_idx:u32) u32
        +row_index(node_id:mesh_NodeId, field_idx:u32) Option~u32~
        +col_index(node_id:mesh_NodeId, field_idx:u32) Option~u32~
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class containers_SubElementField {
        -fespace: Handle~finite_element_space_SubFiniteElementSpace~
        -components: Vec~String~
        -n_cells: usize
        -n_gauss: usize
        -values: Vec~f64~
        +new(fespace:Handle~finite_element_space_SubFiniteElementSpace~, components:Vec~String~) Result~Self~
        +from_uniform_per_component(fespace:Handle~finite_element_space_SubFiniteElementSpace~, components:Vec~String~, values_per_component:&[f64]) Result~Self~
        +fespace() Handle~finite_element_space_SubFiniteElementSpace~
        +cell_count() usize
        +gauss_count() usize
        +component_count() usize
        +components() &[String]
        +component_index(name:&str) Option~usize~
        +get(cell:usize, gauss:usize, comp:usize) Result~f64~
        +set(cell:usize, gauss:usize, comp:usize, value:f64) Result~()~
        +point_values(cell:usize, gauss:usize) Result~&[f64]~
        +value(cell:usize, gauss:usize, component:&str) Result~f64~
        +set_value(cell:usize, gauss:usize, component:&str, value:f64) Result~()~
        +set_uniform(component:&str, value:f64) Result~()~
        +set_cell_uniform(cell:usize, component:&str, value:f64) Result~()~
        +add_to_component(component:&str, scalar:f64) Result~()~
        +sub_to_component(component:&str, scalar:f64) Result~()~
        +mul_to_component(component:&str, scalar:f64) Result~()~
        +div_to_component(component:&str, scalar:f64) Result~()~
        +fill_component(comp:usize, value:f64) void
        +linear_index(cell:usize, gauss:usize, comp:usize) Result~usize~
        +check_cell(cell:usize) Result~()~
        +check_gauss(gauss:usize) Result~()~
        +check_comp(comp:usize) Result~()~
        +component_index_or_err(component:&str) Result~usize~
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +add(rhs:f64) containers_SubElementField
        +sub(rhs:f64) containers_SubElementField
        +mul(rhs:f64) containers_SubElementField
        +div(rhs:f64) containers_SubElementField
    }
    class containers_Clone {
        +clone() Self
    }
    class containers_ElementField {
        -subfields: Vec~Handle<SubElementField>~
        +new(fespace:&FiniteElementSpace, components:Vec~String~) Result~Self~
        +with(fespace:&FiniteElementSpace, components_per_subspace:&[Vec~String~) Result~Self~
        +subfield_count() usize
        +subfield(i:usize) Result~Handle<SubElementField>~
        +index(idx:usize) &Self::Output
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class containers_Aggregate {
        +items() &[Handle~containers_SubElementField~
        +items_mut() &mut Vec~Handle<SubElementField>~
    }
    class containers_IntoIterator {
        +into_iter() Self::IntoIter
    }
    class containers_Physics {
        +primal_vars() Vec~String~
        +dual_vars() Vec~String~
    }
    class containers_SubModel {
        -physics: containers_Physics
        +new(physics:containers_Physics) Self
        +heat_conduction(fespace:Handle~finite_element_space_SubFiniteElementSpace~, material:Handle~containers_SubElementField~) Self
        +dirichlet(config:Handle~mesh_Configuration~, primal_var:String, primal_dual:String, constrained_nodes:Vec~mesh_NodeId~) Result~Self~
        +physics() &Physics
        +multiplier_nodes() &[NodeId]
        +primal_vars() Vec~String~
        +dual_vars() Vec~String~
        +assemble_stiffness(k:&mut Matrix) Result~()~
        +assemble_mass(_m:&mut Matrix) Result~()~
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class containers_Model {
        -sub_models: Vec~Handle<SubModel>~
        +new() Self
        +add_sub_model(sub:Handle~containers_SubModel~) Result~()~
        +sub_model_count() usize
        +sub_model(i:usize) Result~Handle<SubModel>~
        +primal_vars() Result~Vec<String>~
        +dual_vars() Result~Vec<String>~
        +stiffness() Result~containers_Matrix~
        +mass() Result~containers_Matrix~
        +merge(other:&Model) containers_Model
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class containers_Aggregate {
        +items() &[Handle~containers_SubModel~
        +items_mut() &mut Vec~Handle<SubModel>~
    }
    class containers_NodeField {
        -support: Handle~mesh_SubMesh~
        -nodes: Vec~mesh_NodeId~
        -components: Vec~String~
        -values: Vec~f64~
        +from_poi1(submesh:&Handle~mesh_SubMesh~, components:Vec~String~) Result~Self~
        +node_count() usize
        +component_count() usize
        +components() &[String]
        +configuration() Handle~mesh_Configuration~
        +support() Handle~mesh_SubMesh~
        +get(node_idx:usize, comp_idx:usize) Result~f64~
        +set(node_idx:usize, comp_idx:usize, value:f64) Result~()~
        +get_by_node(nid:mesh_NodeId, comp_idx:usize) Result~f64~
        +set_by_node(nid:mesh_NodeId, comp_idx:usize, value:f64) Result~()~
        +index_of(nid:mesh_NodeId) Option~usize~
        +component_index(name:&str) Option~usize~
        +to_poi1_submesh() Result~mesh_SubMesh~
        +to_poi1_mesh() Result~mesh_Mesh~
        +node_values(node_idx:usize) Result~&[f64]~
        +new_with_nodes(cfg:Handle~mesh_Configuration~, nodes:Vec~mesh_NodeId~, components:Vec~String~) Result~Self~
        +component_value_opt(nid:mesh_NodeId, comp:&str) Option~f64~
        +get_or_default(nid:mesh_NodeId, comp:&str) f64
        +check_compatible(other:&NodeField) Result~()~
        +union_layout(other:&NodeField) (Vec~String>, Vec<NodeId~
        +value(nid:mesh_NodeId, component:&str) Result~f64~
        +set_value(nid:mesh_NodeId, component:&str, value:f64) Result~()~
        +add_fields(other:&NodeField) Result~containers_NodeField~
        +merge_fields(other:&NodeField) Result~containers_NodeField~
        +add_to_component(component:&str, scalar:f64) Result~()~
        +sub_to_component(component:&str, scalar:f64) Result~()~
        +mul_to_component(component:&str, scalar:f64) Result~()~
        +div_to_component(component:&str, scalar:f64) Result~()~
        +restrict(mesh:&Mesh) Result~containers_NodeField~
        +check_indices(ni:usize, ci:usize) Result~()~
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +index() &f64
        +index_mut() &mut f64
        +add(rhs:f64) containers_NodeField
        +sub(rhs:f64) containers_NodeField
        +mul(rhs:f64) containers_NodeField
        +div(rhs:f64) containers_NodeField
    }
    class containers_Clone {
        +clone() Self
    }
    class models_Built {
        +constrained_support: Handle~mesh_SubMesh~
        +multiplier_support: Handle~mesh_SubMesh~
        +multiplier_nodes: Vec~mesh_NodeId~
    }
    class models_CellSnapshot {
        -node_ids: Vec~mesh_NodeId~
        -dn_dx: Vec~Vec<f64>~
        -det_j_w: Vec~f64~
    }
    class ops_mesher_triangulation_RefinementOptions {
        +max_edge_length: Option~f64~
        +min_angle_deg: Option~f64~
        +is_active() bool
    }
    class ops_mesher_triangulation_Triangle {
        -v: [usize; 3]
        -n: [usize; 3]
        -alive: bool
    }
    class ops_mesher_triangulation_Cdt {
        -points: Vec~Point2~
        -n_input: usize
        -triangles: Vec~ops_mesher_triangulation_Triangle~
        +new(input_points:&[Point2]) Self
        +insert_point(p_idx:usize) Result~()~
        +rebuild_neighbours() void
        +insert_constraint(a:usize, b:usize) Result~()~
        +edge_exists(a:usize, b:usize) bool
        +triangles_crossing_segment(a:usize, b:usize) Result~Vec<usize>~
        +insert_point_constrained(p_idx:usize, constrained_edges:&HashSet~(usize, usize)~) Result~()~
        +triangle_circumcenter(t_idx:usize) Option~Point2~
        +triangle_longest_edge_sq(t_idx:usize) f64
        +triangle_shortest_edge_sq(t_idx:usize) f64
        +triangle_circumradius_sq(t_idx:usize) f64
        +find_bad_interior_triangle(outside:&[bool], max_edge_sq:Option~f64~, radius_ratio_sq_threshold:Option~f64~) Option~usize~
        +first_encroached_constraint(constrained_edges:&HashSet~(usize, usize)~) Option~(usize, usize)~
        +constraint_has_encroaching_point(a:usize, b:usize, extra_point:Option~Point2~) bool
        +encroached_constraint_by(p:Point2, constrained_edges:&HashSet~(usize, usize)~) Option~(usize, usize)~
        +split_constraint(a:usize, b:usize, constrained_edges:&mut HashSet~(usize, usize)~) Result~usize~
        +refine(opts:&RefinementOptions, constrained_edges:&mut HashSet~(usize, usize)~) Result~()~
        +extract_input_triangles() Vec~[usize; 3]~
        +flood_fill_outside(constrained_edges:&std::collections::HashSet~(usize, usize)~) Vec~bool~
        +extract_interior_with_constraints(constrained_edges:&std::collections::HashSet~(usize, usize)~) Vec~[usize; 3]~
    }
    class mesh_Cell {
        -sm: Handle~mesh_SubMesh~
        -idx: usize
        +new(sm:Handle~mesh_SubMesh~, idx:usize) Result~Self~
        +index() usize
        +element_type() Result~mesh_ElementType~
        +nodes_per_cell() Result~usize~
        +node_ids() Result~Vec<NodeId>~
        +nodes() Result~Vec<Node>~
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class mesh_CellIter {
        -sm: Handle~mesh_SubMesh~
        -next: usize
        -end: usize
        +new(sm:Handle~mesh_SubMesh~, end:usize) Self
    }
    class mesh_Iterator {
        +next() Option~mesh_Cell~
        +size_hint() (usize, Option~usize~
    }
    class mesh_ExactSizeIterator {
    }
    class mesh_RgbColor {
        +r: u8
        +g: u8
        +b: u8
        +new(r:u8, g:u8, b:u8) Self
        +default_face() Self
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class mesh_Default {
        +default() Self
    }
    class mesh_NodeId {
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class mesh_Configuration {
        -dim: u8
        -coord_sets: Vec~Vec<f64>~
        -set_names: Vec~String~
        -active: usize
        -alive: Vec~bool~
        -refcount: Vec~u32~
        -permutation: Option~Vec<u32>~
        +new(dim:u8) Result~Self~
        +dim() u8
        +node_count() usize
        +capacity() usize
        +is_alive(id:mesh_NodeId) bool
        +add_node(coords:&[f64]) Result~mesh_NodeId~
        +incref(id:mesh_NodeId) Result~()~
        +decref(id:mesh_NodeId) Result~()~
        +refcount(id:mesh_NodeId) u32
        +gc() usize
        +coord(id:mesh_NodeId) Result~&[f64]~
        +set_coord(id:mesh_NodeId, coords:&[f64]) Result~()~
        +ensure_alive(id:mesh_NodeId) Result~()~
        +iter_live() impl Iterator~Item = NodeId~
        +add_coord_set(name:impl Into~String~) usize
        +switch_to(set:usize) Result~()~
        +active_set() usize
        +set_names() &[String]
        +permutation() Option~&[u32]~
        +set_permutation(perm:Vec~u32~) Result~()~
        +clear_permutation() void
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class mesh_ElementType {
        +nodes_per_cell() usize
        +topological_dim() usize
        +name() &'static str
        +from_name(s:&str) Option~Self~
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class mesh_Node {
        -handle: Handle~mesh_Configuration~
        -id: mesh_NodeId
        +create_in(cfg:Handle~mesh_Configuration~, coords:&[f64]) Result~Self~
        +acquire(cfg:Handle~mesh_Configuration~, id:mesh_NodeId) Result~Self~
        +from_parts(handle:Handle~mesh_Configuration~, id:mesh_NodeId) Self
        +id() mesh_NodeId
        +configuration() Handle~mesh_Configuration~
        +coord() Result~Vec<f64>~
        +set_coord(coords:&[f64]) Result~()~
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class mesh_Clone {
        +clone() Self
    }
    class mesh_Drop {
        +drop() void
    }
    class mesh_SubMesh {
        -element_type: mesh_ElementType
        -config: Handle~mesh_Configuration~
        -connectivity: Vec~mesh_NodeId~
        -face_color: mesh_RgbColor
        +new(config:Handle~mesh_Configuration~, element_type:mesh_ElementType) Self
        +face_color() mesh_RgbColor
        +set_face_color(color:mesh_RgbColor) void
        +add_cell(nodes:&[NodeId]) Result~usize~
        +add_cell_taking(nodes:&[NodeId]) Result~usize~
        +element_type() mesh_ElementType
        +cell_count() usize
        +connectivity() &[NodeId]
        +configuration() Handle~mesh_Configuration~
        +plot(view:Option~crate::viz::View~, save:Option~&std::path::Path~) Result~()~
        +plot_with_field(view:Option~crate::viz::View~, save:Option~&std::path::Path~, field:&crate::containers::node_field::NodeField, component:Option~&str~) Result~()~
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class mesh_Drop {
        +drop() void
    }
    class mesh_Mesh {
        -submeshes: Vec~Handle<SubMesh>~
        +empty() Self
        +add_submesh(sm:Handle~mesh_SubMesh~) Result~()~
        +submesh_count() usize
        +submesh(idx:usize) Result~Handle<SubMesh>~
        +cell_count() Result~usize~
        +configuration() Result~Handle<Configuration>~
        +with_element_type(config:Handle~mesh_Configuration~, element_type:mesh_ElementType) Self
        +add_cell(nodes:&[NodeId]) Result~usize~
        +element_types() Result~Vec<ElementType>~
        +cell_counts() Result~Vec<usize>~
        +node(submesh_idx:usize, cell_idx:usize, node_idx:usize) Result~mesh_Node~
        +cell(submesh_idx:usize, cell_idx:usize) Result~crate::mesh::cell::Cell~
        +cells(submesh_idx:usize) Result~crate::mesh::cell::CellIter~
        +from_live_nodes(config:Handle~mesh_Configuration~) Result~mesh_Mesh~
        +line_seg2(a:&Node, b:&Node, n_elems:usize) Result~mesh_Mesh~
        +circle_seg2(center:&Node, normal:&[f64], radius:f64, n_elems:usize) Result~mesh_Mesh~
        +sweep_qua4(mesh_a:&Mesh, mesh_b:&Mesh, n_layers:usize) Result~mesh_Mesh~
        +extrude(mesh:&Mesh, direction:&[f64], n_layers:usize) Result~mesh_Mesh~
        +fill_surface(contour:&Mesh, element_type:mesh_ElementType, refinement:Option~crate::ops::mesher::triangulation::RefinementOptions~) Result~mesh_Mesh~
        +plot(view:Option~crate::viz::View~, save:Option~&std::path::Path~) Result~()~
        +plot_with_field(view:Option~crate::viz::View~, save:Option~&std::path::Path~, field:&crate::containers::node_field::NodeField, component:Option~&str~) Result~()~
        +consolidate() Result~mesh_Mesh~
        +merge(other:&Mesh) mesh_Mesh
        +index(idx:usize) &Self::Output
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class mesh_Aggregate {
        +items() &[Handle~mesh_SubMesh~
        +items_mut() &mut Vec~Handle<SubMesh>~
    }
    class mesh_Projection3D {
        -origin: Point3
        -u: Vector3
        -v: Vector3
    }
    class mesh_IntoIterator {
        +into_iter() Self::IntoIter
    }
    class finite_element_space_Interpolation {
        +is_compatible_with(element_type:mesh_ElementType) bool
        +name() &'static str
        +from_name(s:&str) Option~Self~
        +shape(element_type:mesh_ElementType, xi:&[f64]) Result~Vec<f64>~
        +dshape_dxi(element_type:mesh_ElementType, xi:&[f64]) Result~Vec<f64>~
        +check_compat(element_type:mesh_ElementType) Result~()~
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class finite_element_space_QuadratureRule {
        +name() &'static str
        +from_name(s:&str) Option~Self~
        +is_compatible_with(element_type:mesh_ElementType) bool
        +point_count(element_type:mesh_ElementType) Result~usize~
        +points(element_type:mesh_ElementType) Result~(Vec<f64>, Vec<f64>)~
        +check_compat(element_type:mesh_ElementType) Result~()~
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class finite_element_space_SubFiniteElementSpace {
        -submesh: Handle~mesh_SubMesh~
        -interpolation: finite_element_space_Interpolation
        -quadrature: finite_element_space_QuadratureRule
        -space_dim: usize
        -gauss_xi: Vec~f64~
        -gauss_w: Vec~f64~
        -n_at_g: Vec~f64~
        -dn_at_g: Vec~f64~
        +new(submesh:Handle~mesh_SubMesh~, interpolation:finite_element_space_Interpolation, quadrature:finite_element_space_QuadratureRule) Result~Self~
        +submesh() Handle~mesh_SubMesh~
        +configuration() Result~Handle<Configuration>~
        +interpolation() finite_element_space_Interpolation
        +quadrature() finite_element_space_QuadratureRule
        +element_type() Result~mesh_ElementType~
        +ref_dim() Result~usize~
        +space_dim() usize
        +nodes_per_cell() Result~usize~
        +cell_count() Result~usize~
        +gauss_count() usize
        +gauss_xi(g:usize) Result~&[f64]~
        +gauss_weight(g:usize) Result~f64~
        +n_at_g(g:usize) Result~&[f64]~
        +dn_at_g(g:usize) Result~&[f64]~
        +jacobian(cell_idx:usize, g:usize) Result~Vec<f64>~
        +det_jacobian(cell_idx:usize, g:usize) Result~f64~
        +dn_dx(cell_idx:usize, g:usize) Result~Vec<f64>~
        +cell_node_coords(cell_idx:usize) Result~Vec<f64>~
        +check_g(g:usize) Result~()~
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class finite_element_space_FiniteElementSpace {
        -subspaces: Vec~Handle<SubFiniteElementSpace>~
        +with(mesh:&Mesh, choices:&[(Interpolation, QuadratureRule)]) Result~Self~
        +new(mesh:&Mesh, interpolation:finite_element_space_Interpolation) Result~Self~
        +lagrange1(mesh:&Mesh) Result~Self~
        +subspace_count() usize
        +subspace(i:usize) Result~Handle<SubFiniteElementSpace>~
        +merge(other:&FiniteElementSpace) finite_element_space_FiniteElementSpace
        +index(idx:usize) &Self::Output
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
    }
    class finite_element_space_Aggregate {
        +items() &[Handle~finite_element_space_SubFiniteElementSpace~
        +items_mut() &mut Vec~Handle<SubFiniteElementSpace>~
    }
    class finite_element_space_IntoIterator {
        +into_iter() Self::IntoIter
    }
    class PyrucastError {
        +fmt(f:&mut fmt::Formatter~'_~) fmt::Result
        +from(e:std::io::Error) Self
    }
    class Persist {
        +to_bytes() Result~Vec<u8>~
        +from_bytes(bytes:&[u8]) Result~Self~
    }
    class Slot {
        -state: SlotState~T~
        -gen: u32
        -refcount: u32
    }
    class StoreInner {
        -slots: Vec~Slot<T>~
        -free: Vec~u32~
    }
    class Handle {
        -idx: u32
        -gen: u32
        -_t: PhantomData~fn() -> T~
    }
    class Clone {
        +clone() Self
    }
    class Drop {
        +drop() void
    }
namespace tests {
    class tests_PInsertGet {
    }
    class tests_PClone {
    }
    class tests_PRecycle {
    }
    class tests_PStale {
    }
    class tests_PMut {
    }
    class tests_PSwap {
    }
    class tests_PCompact {
    }
    class tests_PDisplay {
    }
    class tests_PSwapDrop {
    }
    class tests_Drop {
        +drop() void
    }
    class tests_Item {
    }
    class tests_Bag {
        -items: Vec~Handle<Item>~
    }
    class tests_Aggregate {
        +items() &[Handle~tests_Item~
        +items_mut() &mut Vec~Handle<Item>~
    }
}
    viz_Drawable ..> viz_Bbox3
    viz_Drawable ..> viz_Bbox3
    viz_Drawable ..> viz_Bbox3
    viz_Drawable ..> viz_Bbox3
    viz_FieldDrawable --> mesh_Cell
    viz_Drawable ..> viz_Bbox3
    py_PyCell --> mesh_Cell
    py_PyCell ..> py_PyNode
    py_PyConfiguration --> Handle
    py_PyConfiguration --> mesh_Configuration
    py_PyConfiguration ..> py_PyNode
    py_PyConfiguration ..> py_PyNode
    py_PySubElementField --> Handle
    py_PySubElementField --> containers_SubElementField
    py_PySubElementField ..> py_PySubElementField
    py_PySubElementField ..> py_PySubElementField
    py_PySubElementField ..> py_PySubElementField
    py_PySubElementField ..> py_PySubElementField
    py_PyElementField --> containers_ElementField
    py_PyElementField ..> py_PySubElementField
    py_PySubFiniteElementSpace --> Handle
    py_PySubFiniteElementSpace --> finite_element_space_SubFiniteElementSpace
    py_PyFiniteElementSpace --> finite_element_space_FiniteElementSpace
    py_PyFiniteElementSpace ..> py_PySubFiniteElementSpace
    py_PyMatrix --> Handle
    py_PyMatrix --> containers_Matrix
    py_PySubModel --> Handle
    py_PySubModel --> containers_SubModel
    py_PyModel --> containers_Model
    py_PyModel ..> py_PySubModel
    py_PyModel ..> py_PyMatrix
    py_PyModel ..> py_PyMatrix
    py_PyNode --> mesh_Node
    py_PyNodeField --> Handle
    py_PyNodeField --> containers_NodeField
    py_PyNodeField ..> py_PySubMesh
    py_PyNodeField ..> py_PyMesh
    py_PyNodeField ..> py_PyNodeField
    py_PyNodeField ..> py_PyNodeField
    py_PyNodeField ..> py_PyNodeField
    py_PyNodeField ..> py_PyNodeField
    py_PyNodeField ..> py_PyNodeField
    py_PyNodeField ..> py_PyNodeField
    py_PyNodeField ..> py_PyNodeField
    py_PySubMesh --> Handle
    py_PySubMesh --> mesh_SubMesh
    py_PyMesh --> mesh_Mesh
    py_PyMesh ..> py_PyNode
    py_PyMesh ..> py_PyMesh
    py_PyMesh ..> py_PyMesh
    py_PyMesh ..> py_PySubMesh
    containers_DofId --> mesh_NodeId
    containers_Matrix --> containers_DofId
    containers_Matrix --> containers_DofId
    containers_Matrix ..> containers_DofId
    containers_SubElementField --> Handle
    containers_SubElementField --> finite_element_space_SubFiniteElementSpace
    containers_SubElementField ..> Handle
    containers_SubElementField ..> finite_element_space_SubFiniteElementSpace
    containers_SubElementField ..> containers_SubElementField
    containers_SubElementField ..> containers_SubElementField
    containers_SubElementField ..> containers_SubElementField
    containers_SubElementField ..> containers_SubElementField
    containers_Aggregate ..> containers_SubElementField
    containers_SubModel --> containers_Physics
    containers_Model ..> containers_Matrix
    containers_Model ..> containers_Matrix
    containers_Model ..> containers_Model
    containers_Aggregate ..> containers_SubModel
    containers_NodeField --> Handle
    containers_NodeField --> mesh_SubMesh
    containers_NodeField --> mesh_NodeId
    containers_NodeField ..> Handle
    containers_NodeField ..> mesh_Configuration
    containers_NodeField ..> Handle
    containers_NodeField ..> mesh_SubMesh
    containers_NodeField ..> mesh_SubMesh
    containers_NodeField ..> mesh_Mesh
    containers_NodeField ..> containers_NodeField
    containers_NodeField ..> containers_NodeField
    containers_NodeField ..> containers_NodeField
    containers_NodeField ..> containers_NodeField
    containers_NodeField ..> containers_NodeField
    containers_NodeField ..> containers_NodeField
    containers_NodeField ..> containers_NodeField
    models_Built --> Handle
    models_Built --> mesh_SubMesh
    models_Built --> Handle
    models_Built --> mesh_SubMesh
    models_Built --> mesh_NodeId
    models_CellSnapshot --> mesh_NodeId
    ops_mesher_triangulation_Cdt --> ops_mesher_triangulation_Triangle
    mesh_Cell --> Handle
    mesh_Cell --> mesh_SubMesh
    mesh_Cell ..> mesh_ElementType
    mesh_CellIter --> Handle
    mesh_CellIter --> mesh_SubMesh
    mesh_Iterator ..> mesh_Cell
    mesh_Configuration ..> mesh_NodeId
    mesh_Node --> Handle
    mesh_Node --> mesh_Configuration
    mesh_Node --> mesh_NodeId
    mesh_Node ..> mesh_NodeId
    mesh_Node ..> Handle
    mesh_Node ..> mesh_Configuration
    mesh_SubMesh --> mesh_ElementType
    mesh_SubMesh --> Handle
    mesh_SubMesh --> mesh_Configuration
    mesh_SubMesh --> mesh_NodeId
    mesh_SubMesh --> mesh_RgbColor
    mesh_SubMesh ..> mesh_RgbColor
    mesh_SubMesh ..> mesh_ElementType
    mesh_SubMesh ..> Handle
    mesh_SubMesh ..> mesh_Configuration
    mesh_Mesh ..> mesh_Node
    mesh_Mesh ..> mesh_Mesh
    mesh_Mesh ..> mesh_Mesh
    mesh_Mesh ..> mesh_Mesh
    mesh_Mesh ..> mesh_Mesh
    mesh_Mesh ..> mesh_Mesh
    mesh_Mesh ..> mesh_Mesh
    mesh_Mesh ..> mesh_Mesh
    mesh_Mesh ..> mesh_Mesh
    mesh_Aggregate ..> mesh_SubMesh
    finite_element_space_SubFiniteElementSpace --> Handle
    finite_element_space_SubFiniteElementSpace --> mesh_SubMesh
    finite_element_space_SubFiniteElementSpace --> finite_element_space_Interpolation
    finite_element_space_SubFiniteElementSpace --> finite_element_space_QuadratureRule
    finite_element_space_SubFiniteElementSpace ..> Handle
    finite_element_space_SubFiniteElementSpace ..> mesh_SubMesh
    finite_element_space_SubFiniteElementSpace ..> finite_element_space_Interpolation
    finite_element_space_SubFiniteElementSpace ..> finite_element_space_QuadratureRule
    finite_element_space_SubFiniteElementSpace ..> mesh_ElementType
    finite_element_space_FiniteElementSpace ..> finite_element_space_FiniteElementSpace
    finite_element_space_Aggregate ..> finite_element_space_SubFiniteElementSpace
    tests_Aggregate ..> tests_Item

```