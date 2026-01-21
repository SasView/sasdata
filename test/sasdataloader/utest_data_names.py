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
    ("ISIS_1_1", "TK49 c10_SANS:79680:Q:4TghWEoJi6xxhyeDXhS751"),
    ("cansas1d", "Test title:1234:Q:440tNBqdx9jvci6CgjmrmD"),
    ("MAR07232_rest", "MAR07232_rest_out.dat:2:/sasentry01/sasdata01/Qx:2Y0qTTb054KSJnJaJv0rFl"),
    ("simpleexamplefile", "::/sasentry01/sasdata01/Q:uoHMeB8mukElC1uLCy7Sd"),
]


@pytest.mark.names
@pytest.mark.parametrize("x", test_file_names)
def test_quantity_name(x):
    (f, expected) = x
    data = [v for v in local_load(f"data/{f}").values()][0]
    if data.metadata.title is not None:
        assert data.abscissae.unique_id.startswith(data.metadata.title)
    assert data.abscissae.unique_id == expected
