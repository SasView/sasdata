import numpy as np

from sasdata.quantities.quantity import Quantity
from sasdata.quantities.units import per_angstrom, radians
from sasdata.slicing.meshes.mesh import Mesh
from sasdata.slicing.rebinning import Rebinner

QuantityType = Quantity | float


class AnnularSector(Rebinner):
    """ A single annular sector (wedge sum)"""
    def __init__(self, q0: QuantityType, q1: QuantityType, phi0: QuantityType,
                 phi1: QuantityType, points_per_degree: int=2):
        super().__init__()

        # Ensure all values are scaled to the proper units
        self._q0 = q0.to_units_of(per_angstrom) if isinstance(q0, Quantity) else Quantity(q0, per_angstrom)
        self._q1 = q1.to_units_of(per_angstrom) if isinstance(q1, Quantity) else Quantity(q1, per_angstrom)
        self._phi0 = phi0.to_units_of(radians) if isinstance(phi0, Quantity) else Quantity(phi0, radians)
        self._phi1 = phi1.to_units_of(radians) if isinstance(phi1, Quantity) else Quantity(phi1, radians)

        self.points_per_degree = points_per_degree

    @property
    def q0(self):
        return self._q0.value

    @property
    def q1(self):
        return self._q1.value

    @property
    def phi0(self):
        return self._phi0.value

    @property
    def phi1(self):
        return self._phi1.value

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
    AnnularSector(1, 2, 1, 2).bin_mesh.show()


if __name__ == "__main__":
    main()
