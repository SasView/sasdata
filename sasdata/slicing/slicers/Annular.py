from sasdata.quantities.constants import TwoPi
from sasdata.quantities.quantity import Quantity
from sasdata.quantities.units import per_angstrom
from sasdata.slicing.slicers.AnnularSector import AnnularSector, QuantityType


class Annular(AnnularSector):
    """ Annual averaging, using the wedge as a basis. """
    def __init__(self, q0: QuantityType, q1: QuantityType, points_per_degree: int=2):
        super().__init__(q0, q1, 0.0, TwoPi, points_per_degree)


def main():
    """ Just show a random example"""
    a_float = Annular(1, 2)
    q0 = Quantity(1, per_angstrom)
    q1 = Quantity(2, per_angstrom)
    a_quant = Annular(q0, q1)
    assert a_float.bin_mesh.n_cells == a_quant.bin_mesh.n_cells


if __name__ == "__main__":
    main()
