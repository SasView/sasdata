import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.spatial import Voronoi

# Some test data

qx_base_values = np.linspace(-10, 10, 21)
qy_base_values = np.linspace(-10, 10, 21)

qx, qy = np.meshgrid(qx_base_values, qy_base_values)

include = np.logical_not((np.abs(qx) < 2) & (np.abs(qy) < 2))

qx = qx[include]
qy = qy[include]

r = np.sqrt(qx**2 + qy**2)

data = np.log((1+np.cos(3*r))*np.exp(-r*r))

colormap = cm.get_cmap('winter', 256)

def get_data_mesh(x, y, data):

    input_data = np.array((x, y)).T
    voronoi = Voronoi(input_data)

    # plt.scatter(voronoi.vertices[:,0], voronoi.vertices[:,1])
    # plt.scatter(voronoi.points[:,0], voronoi.points[:,1])

    cmin = np.min(data)
    cmax = np.max(data)

    color_index_map = np.array(255 * (data - cmin) / (cmax - cmin), dtype=int)

    for point_index, points in enumerate(voronoi.points):

        region_index = voronoi.point_region[point_index]
        region = voronoi.regions[region_index]

        if len(region) > 0:

            if -1 in region:

                pass

            else:

                color = colormap(color_index_map[point_index])

                circly = region + [region[0]]
                plt.fill(voronoi.vertices[circly, 0], voronoi.vertices[circly, 1], color=color, edgecolor="white")

    plt.show()

get_data_mesh(qx.reshape(-1), qy.reshape(-1), data)