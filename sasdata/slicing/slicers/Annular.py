from sasdata.quantities.constants import TwoPi
from sasdata.slicing.slicers.AnnularSector import AnnularSector


class Annular(AnnularSector):
    """ Annual averaging, using the wedge as a basis. """
    def __init__(self, q0: float, q1: float, points_per_degree: int=2):
        super().__init__(q0, q1, 0.0, TwoPi, points_per_degree)


def main():
    """ Just show a random example"""
    Annular(1, 2).bin_mesh.show()


if __name__ == "__main__":
    main()
