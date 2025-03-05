import pytest
import numpy as np

from sasdata.data import SasData
from sasdata.dataset_types import one_dim
from sasdata.data_backing import Group
from sasdata.quantities.quantity import Quantity
from sasdata.quantities.units import per_angstrom, per_centimeter

def test_1d():
    q = [1, 2, 3, 4, 5]
    i = [5, 4, 3, 2, 1]

    q_quantity = Quantity(np.array(q), per_angstrom)
    i_quantity = Quantity(np.array(i), per_centimeter)

    data_contents = {
        'Q': q_quantity,
        'I': i_quantity
    }

    data = SasData('TestData', data_contents, one_dim, Group('root', {}), True)

    assert all(data.abscissae.value == np.array(q_quantity))
    assert all(data.ordinate.value == np.array(i_quantity))
