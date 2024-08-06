

class DistributionModel:


    @property
    def is_density(self) -> bool:
        return False

    def standard_deviation(self) -> Quantity:
        return NotImplementedError("Variance not implemented yet")
