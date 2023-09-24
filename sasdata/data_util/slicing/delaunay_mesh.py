import numpy as np

from scipy.spatial import Delaunay

from sasdata.data_util.slicing.mesh import Mesh


def delaunay_mesh(x, y) -> Mesh:

    input_data = np.array((x, y)).T
    delaunay = Delaunay(input_data)

    edges = set()

    for simplex_index, simplex in enumerate(delaunay.simplices):

        wrapped = list(simplex) + [simplex[0]]

        for a, b in zip(wrapped[:-1], wrapped[1:]):
            # make sure the representation is unique
            if a > b:
                edges.add((a, b))
            else:
                edges.add((b, a))

    edges = list(edges)

    return Mesh(points=input_data, edges=edges, cells=[])


if __name__ == "__main__":
    points = np.random.random((100, 2))
    mesh = delaunay_mesh(points[:,0], points[:,1])
    mesh.show()