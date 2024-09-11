from sasdata.quantities.unit_parser import parse_unit
from sasdata.quantities.units_tests import EqualUnits
from sasdata.quantities.units import Dimensions, Unit, meters, meters_per_second, per_angstrom, \
    kilometers_per_square_hour

# Lets start with the straight forward ones first, and get progressivel more complex as the list goes on.
tests = [
    EqualUnits('Metres',
               meters,
               parse_unit('m')),
    EqualUnits('Metres per second',
               meters_per_second,
               parse_unit('ms-1'),
               parse_unit('m/s')),
    EqualUnits('Inverse Test',
               per_angstrom,
               parse_unit('1/A'),
               parse_unit('A-1')),
    # This test is primarily to ensure that the 'mm' doesn't get interpreted as two separate metres.
    EqualUnits('Milimetres * Centimetres',
               # TODO: Not sure if this calculation is right.
               Unit(0.001 * 0.01, Dimensions(length=2)),
               parse_unit('mmcm')),
    EqualUnits("Acceleration",
               kilometers_per_square_hour,
               parse_unit('kmh-2'),
               parse_unit('km/h2')
               )
]

for test in tests:
    print(test.test_name)
    test.run_test()
