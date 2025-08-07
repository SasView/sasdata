import numpy as np
from scipy.spatial import Delaunay

from sasdata.slicing.meshes.mesh import Mesh


def delaunay_mesh(x, y) -> Mesh:
    """ Create a triangulated mesh based on input points """

    input_data = np.array((x, y)).T
    delaunay = Delaunay(input_data)

    return Mesh(points=input_data, cells=delaunay.simplices)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    points = np.random.random((100, 2))
    mesh = delaunay_mesh(points[:,0], points[:,1])
    mesh.show(actually_show=False)

    print(mesh.cells[50])

    # pick random cell to show
    for cell in mesh.cells_to_edges[10]:
        a, b = mesh.edges[cell]
        plt.plot(
            [mesh.points[a][0], mesh.points[b][0]],
            [mesh.points[a][1], mesh.points[b][1]],
            color='r')

    plt.show()
