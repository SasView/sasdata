import numpy as np

from sasdata.data_util.slicing.rebinning import Rebinner
from sasdata.data_util.slicing.meshes.mesh import Mesh

class AnularSector(Rebinner):
    """ A single annular sector (wedge sum)"""
    def __init__(self, q0: float, q1: float, phi0: float, phi1: float, order: int=1, points_per_degree: int=2):
        super().__init__(order)

        self.q0 = q0
        self.q1 = q1
        self.phi0 = phi0
        self.phi1 = phi1

        self.points_per_degree = points_per_degree

    def _bin_mesh(self) -> Mesh:

        n_points = 1 + 180*self.points_per_degree*(self.phi1 - self.phi0) / np.pi

        angles = np.linspace(self.phi0, self.phi1, n_points)

        row1 = self.q0 * np.array([np.cos(angles), np.sin(angles)])
        row2 = self.q1 * np.array([np.cos(angles), np.sin(angles)])[:, ::-1]

        points = np.concatenate((row1, row2), axis=1)

        cells = [i for i in range(2*n_points)]

        return Mesh(points=points, cells=cells)

    def _bin_coordinates(self) -> np.ndarray:
        return np.array([], dtype=float)

