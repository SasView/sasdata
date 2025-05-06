"""
Unit tests for the new recursive cansas reader
"""

import os
import pytest


from sasdata.temp_hdf5_reader import load_data
from sasdata.temp_sesans_reader import load_data


def local_load(path: str):
    """Get local file path"""
    return os.path.join(os.path.dirname(__file__), path)


@pytest.mark.sesans
def test_load_file():
    data = load_data(local_load(f"sesans_data/sphere2micron.ses"))

    with open(local_load(f"reference/sphere2micron.txt")) as infile:
        expected = "".join(infile.readlines())
    assert data.summary() == expected
