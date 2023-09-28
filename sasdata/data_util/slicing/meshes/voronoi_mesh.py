import numpy as np
from scipy.spatial import Voronoi


from sasdata.data_util.slicing.meshes.mesh import Mesh

def voronoi_mesh(x, y) -> Mesh:

    input_data = np.array((x.reshape(-1), y.reshape(-1))).T

    # Need to make sure mesh covers a finite region, probably not important for
    # much data stuff, but is important for plotting
    #   To do this first need to find an appropriate region
    #   Then we need to adjust the mesh to deal with these points

    voronoi = Voronoi(input_data)




    finite_cells = [region for region in voronoi.regions if -1 not in region and len(region) > 0]



    return Mesh(points=voronoi.vertices, cells=finite_cells)


if __name__ == "__main__":
    points = np.random.random((100, 2))
    mesh = voronoi_mesh(points[:,0], points[:,1])
    mesh.show()