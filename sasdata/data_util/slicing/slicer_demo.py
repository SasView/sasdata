""" Dev docs: """

import numpy as np

from sasdata.data_util.slicing.slicers import AnularSector
from sasdata.data_util.slicing.meshes.mesh import Mesh
from sasdata.data_util.slicing.meshes.voronoi_mesh import voronoi_mesh



if __name__ == "__main__":

    # Demo of sums, annular sector over some not very circular data

    q_range = 1.5

    test_coordinates = (2*q_range)*(np.random.random((100, 2))-0.5)

    # Demo of averaging, annular sector over ring shaped data