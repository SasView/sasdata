import math

import sasdata.quantities.units as units
from sasdata.quantities.units import Unit

class EqualUnits:
    def __init__(self, test_name: str, *units):
        self.test_name = "Equality: " + test_name
        self.units: list[Unit] = list(units)

    def run_test(self):
        for i, unit_1 in enumerate(self.units):
            for unit_2 in self.units[i + 1 :]:
                assert unit_1.equivalent(unit_2), "Units should be equivalent"
                assert unit_1 == unit_2, "Units should be equal"


class EquivalentButUnequalUnits:
    def __init__(self, test_name: str, *units):
        self.test_name = "Equivalence: " + test_name
        self.units: list[Unit] = list(units)

    def run_test(self):
        for i, unit_1 in enumerate(self.units):
            for unit_2 in self.units[i + 1 :]:
                assert unit_1.equivalent(unit_2), "Units should be equivalent"
                assert unit_1 != unit_2, "Units should not be equal"


class DissimilarUnits:
    def __init__(self, test_name: str, *units):
        self.test_name = "Dissimilar: " + test_name
        self.units: list[Unit] = list(units)

    def run_test(self):
        for i, unit_1 in enumerate(self.units):
            for unit_2 in self.units[i + 1 :]:
                assert not unit_1.equivalent(unit_2), "Units should not be equivalent"


tests = [

    EqualUnits("Pressure",
               units.pascals,
               units.newtons / units.meters ** 2,
               units.micronewtons * units.millimeters ** -2),

    EqualUnits("Resistance",
               units.ohms,
               units.volts / units.amperes,
               1e-3/units.millisiemens),

    EquivalentButUnequalUnits("Angular frequency",
               units.rotations / units.minutes,
               units.degrees * units.hertz),

    EqualUnits("Angular frequency",
               (units.rotations/units.minutes ),
               (units.radians*units.hertz) * 2 * math.pi/60.0),

    DissimilarUnits("Frequency and Angular frequency",
                    (units.rotations/units.minutes),
                    (units.hertz)),


]


for test in tests:
    print(test.test_name)
    test.run_test()
