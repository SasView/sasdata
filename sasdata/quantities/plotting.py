import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

from sasdata.quantities.quantity import NamedQuantity, Quantity


def quantity_plot(x: Quantity[ArrayLike], y: Quantity[ArrayLike], *args, **kwargs):
    plt.plot(x.value, y.value, *args, **kwargs)

    x_name = x.name if isinstance(x, NamedQuantity) else "x"
    y_name = y.name if isinstance(y, NamedQuantity) else "y"

    plt.xlabel(f"{x_name} / {x.units}")
    plt.ylabel(f"{y_name} / {y.units}")

def quantity_scatter(x: Quantity[ArrayLike], y: Quantity[ArrayLike], *args, **kwargs):
    plt.scatter(x.value, y.value, *args, **kwargs)

    x_name = x.name if isinstance(x, NamedQuantity) else "x"
    y_name = y.name if isinstance(y, NamedQuantity) else "y"

    plt.xlabel(f"{x_name} / {x.units}")
    plt.ylabel(f"{y_name} / {y.units}")
