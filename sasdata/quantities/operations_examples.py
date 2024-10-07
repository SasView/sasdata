from sasdata.quantities.operations import Variable, Mul

x = Variable("x")
y = Variable("y")
z = Variable("z")
f = Mul(Mul(x, y), z)


dfdx = f.derivative(x).derivative(y).derivative(z)

print(dfdx.summary())
