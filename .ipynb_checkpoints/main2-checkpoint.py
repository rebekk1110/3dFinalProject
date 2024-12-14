import open3d as o3d

mesh = o3d.geometry.TriangleMesh.create_sphere()
o3d.io.write_triangle_mesh("sphere.ply", mesh)
