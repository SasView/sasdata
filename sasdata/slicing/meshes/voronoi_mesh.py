import numpy as np
from scipy.spatial import Voronoi

from sasdata.slicing.meshes.mesh import Mesh


def voronoi_mesh(x, y, debug_plot=False) -> Mesh:
    """ Create a mesh based on a voronoi diagram of points """

    input_data = np.array((x.reshape(-1), y.reshape(-1))).T

    # Need to make sure mesh covers a finite region, probably not important for
    # much data stuff, but is important for plotting
    #
    # * We want the cells at the edge of the mesh to have a reasonable size, definitely not infinite
    # * The exact size doesn't matter that much
    # * It should work well with a grid, but also
    # * ...it should be robust so that if the data isn't on a grid, it doesn't cause any serious problems
    #
    # Plan: Create a square border of points that are totally around the points, this is
    #       at the distance it would be if it was an extra row of grid points
    #       to do this we'll need
    #       1) an estimate of the grid spacing
    #       2) the bounding box of the grid
    #


    # Use the median area of finite voronoi cells as an estimate
    voronoi = Voronoi(input_data)
    finite_cells = [region for region in voronoi.regions if -1 not in region and len(region) > 0]
    premesh = Mesh(points=voronoi.vertices, cells=finite_cells)

    area_spacing = np.median(premesh.areas)
    gap = np.sqrt(area_spacing)

    # Bounding box is easy
    x_min, y_min = np.min(input_data, axis=0)
    x_max, y_max = np.max(input_data, axis=0)

    # Create a border
    n_x = int(np.round((x_max - x_min)/gap))
    n_y = int(np.round((y_max - y_min)/gap))

    top_bottom_xs = np.linspace(x_min - gap, x_max + gap, n_x + 3)
    left_right_ys = np.linspace(y_min, y_max, n_y + 1)

    top = np.array([top_bottom_xs, (y_max + gap) * np.ones_like(top_bottom_xs)])
    bottom = np.array([top_bottom_xs, (y_min - gap) * np.ones_like(top_bottom_xs)])
    left = np.array([(x_min - gap) * np.ones_like(left_right_ys), left_right_ys])
    right = np.array([(x_max + gap) * np.ones_like(left_right_ys), left_right_ys])

    added_points = np.concatenate((top, bottom, left, right), axis=1).T

    if debug_plot:
        import matplotlib.pyplot as plt
        plt.scatter(x, y)
        plt.scatter(added_points[:, 0], added_points[:, 1])
        plt.show()

    new_points = np.concatenate((input_data, added_points), axis=0)
    voronoi = Voronoi(new_points)

    # Remove the cells that correspond to the added edge points,
    # Because the points on the edge of the square are (weakly) convex, these
    # regions be infinite

    # finite_cells = [region for region in voronoi.regions if -1 not in region and len(region) > 0]

    # ... however, we can just use .region_points
    input_regions = voronoi.point_region[:input_data.shape[0]]
    cells = [voronoi.regions[region_index] for region_index in input_regions]

    return Mesh(points=voronoi.vertices, cells=cells)


def square_grid_check():
    values = np.linspace(-10, 10, 21)
    x, y = np.meshgrid(values, values)

    mesh = voronoi_mesh(x, y)

    mesh.show(show_labels=True)

def random_grid_check():
    import matplotlib.pyplot as plt
    points = np.random.random((100, 2))
    mesh = voronoi_mesh(points[:, 0], points[:, 1], True)
    mesh.show(actually_show=False)
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()


if __name__ == "__main__":
    square_grid_check()
    # random_grid_check()

