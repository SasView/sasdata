from sasdata.quantities.quantity import Quantity, NamedQuantity
from sasdata.quantities import units

x = NamedQuantity("x", 1, units.meters, variance=1)
y = NamedQuantity("y", 1, units.meters, variance=1)

print(x+y)
print(x+x)