"""
Tests for generation of unique, but reproducible, names for data quantities
"""


import pytest

from test import local_data_finder, local_load

test_file_names = [
    ("ascii_test_1", "::Q:3KrS58TPgclJ1rgyr0VQp3"),
    ("ISIS_1_1", "TK49 c10_SANS:79680:Q:4TghWEoJi6xxhyeDXhS751"),
    ("cansas1d", "Test title:1234:Q:440tNBqdx9jvci6CgjmrmD"),
    ("MAR07232_rest", "MAR07232_rest_out.dat:2:/sasentry01/sasdata01/Qx:37t0tPj1o8oQcQEB3DVUlw"),
    ("simpleexamplefile", "::/sasentry01/sasdata01/Q:uoHMeB8mukElC1uLCy7Sd"),
]


@pytest.mark.names
@pytest.mark.parametrize("x", test_file_names)
def test_quantity_name(x):
    (f, expected) = x
    data = [v for v in local_load(local_data_finder(f))][0]
    if data.metadata.title is not None:
        assert data.abscissae.axes[0].unique_id.startswith(data.metadata.title)
    assert data.abscissae.axes[0].unique_id == expected
