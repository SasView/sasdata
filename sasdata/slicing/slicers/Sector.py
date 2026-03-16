from sasdata.quantities.constants import Inf, NegInf, PiOverTwo
from sasdata.slicing.slicers.AnnularSector import AnnularSector


class Sector(AnnularSector):
    """ Annual averaging, using the wedge as a basis. """
    def __init__(self, phi0: float, phi1: float, points_per_degree: int=2):
        super().__init__(NegInf, Inf, phi0, phi1, points_per_degree)


def main():
    """ Just show a random example"""
    sector = Sector(0, PiOverTwo)
    sector.q0 = -3
    sector.q1 = 3
    sector.bin_mesh.show()


if __name__ == "__main__":
    main()
