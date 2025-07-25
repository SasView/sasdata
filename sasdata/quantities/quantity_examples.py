from sasdata.quantities.quantity import NamedQuantity
from sasdata.quantities import units

x = NamedQuantity("x", 1, units.meters, standard_error=1)
y = NamedQuantity("y", 1, units.decimeters, standard_error=1)

print(x+y)
print((x+y).to_units_of(units.centimeters))