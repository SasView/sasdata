import numpy as np

from sasdata.slicing.meshes.mesh import Mesh
from sasdata.slicing.rebinning import Rebinner


class AnularSector(Rebinner):
    """ A single annular sector (wedge sum)"""
    def __init__(self, q0: float, q1: float, phi0: float, phi1: float, points_per_degree: int=2):
        super().__init__()

        self.q0 = q0
        self.q1 = q1
        self.phi0 = phi0
        self.phi1 = phi1

        self.points_per_degree = points_per_degree

    def _bin_mesh(self) -> Mesh:

        n_points = np.max([int(1 + 180*self.points_per_degree*(self.phi1 - self.phi0) / np.pi), 2])

        angles = np.linspace(self.phi0, self.phi1, n_points)

        row1 = self.q0 * np.array([np.cos(angles), np.sin(angles)])
        row2 = self.q1 * np.array([np.cos(angles), np.sin(angles)])[:, ::-1]

        points = np.concatenate((row1, row2), axis=1).T

        cells = [[i for i in range(2*n_points)]]

        return Mesh(points=points, cells=cells)

    def _bin_coordinates(self) -> np.ndarray:
        return np.array([], dtype=float)


def main():
    """ Just show a random example"""
    AnularSector(1, 2, 1, 2).bin_mesh.show()


if __name__ == "__main__":
    main()