"""
Tests for generation of unique, but reproducible, names for data quantities
"""

import os
import pytest

from sasdata.data import SasData
from sasdata.temp_hdf5_reader import load_data as hdf_load_data
from sasdata.temp_xml_reader import load_data as xml_load_data


def local_load(path: str) -> SasData:
    """Get local file path"""
    base = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(f"{base}.h5"):
        return hdf_load_data(f"{base}.h5")
    if os.path.exists(f"{base}.xml"):
        return xml_load_data(f"{base}.xml")
    assert False


test_file_names = [
    ("ISIS_1_1", ":0x7e5a2c481e82056d2d69c005af3eb655"),
    ("cansas1d", ":0xb7d42213981e09c208e32527729f8dc6"),
    ("MAR07232_rest", ":0x53bd3d9e0644c62c00db705b58a69f55"),
    ("simpleexamplefile", ":0x2e9b96e912dd0af39b9f44f139d6fc47"),
]


@pytest.mark.names
@pytest.mark.parametrize("x", test_file_names)
def test_quantity_name(x):
    (f, expected) = x
    data = [v for v in local_load(f"data/{f}").values()][0]
    assert data.abscissae.unique_id == expected
