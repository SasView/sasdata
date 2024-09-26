from sasdata.quantities.quantity import Quantity
from sasdata.quantities import units

x = Quantity(1, units.meters, variance=1)
y = Quantity(1, units.meters, variance=1)

z = x+y

print(z)