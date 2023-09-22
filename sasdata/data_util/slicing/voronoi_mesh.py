import numpy as np
from scipy.spatial import Voronoi


from sasdata.data_util.slicing.mesh import Mesh

def voronoi_mesh(x, y) -> Mesh:

    input_data = np.array((x, y)).T
    voronoi = Voronoi(input_data)

    edges = set()

    for point_index, points in enumerate(voronoi.points):

        region_index = voronoi.point_region[point_index]
        region = voronoi.regions[region_index]

        wrapped = region + [region[0]]
        for a, b in zip(wrapped[:-1], wrapped[1:]):
            if not a == -1 and not b == -1:

                # make sure the representation is unique
                if a > b:
                    edges.add((a, b))
                else:
                    edges.add((b, a))

    edges = list(edges)

    return Mesh(points=voronoi.vertices, edges=edges, cells=[])


if __name__ == "__main__":
    points = np.random.random((100, 2))
    mesh = voronoi_mesh(points[:,0], points[:,1])
    mesh.show()