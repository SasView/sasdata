"""
Unit tests for the new recursive cansas reader
"""

import os

import numpy as np
import pytest

from sasdata.model_requirements import (
    ComposeRequirements,
    ModellingRequirements,
    NullModel,
    PinholeModel,
    SesansModel,
    SlitModel,
    guess_requirements,
)
from sasdata.quantities import unit_parser, units
from sasdata.quantities.quantity import Quantity
from sasdata.temp_sesans_reader import load_data

test_file_names = ["sphere2micron", "sphere_isis"]


def local_load(path: str):
    """Get local file path"""
    return os.path.join(os.path.dirname(__file__), path)


@pytest.mark.sesans
@pytest.mark.parametrize("f", test_file_names)
def test_load_file(f):
    data = load_data(local_load(f"sesans_data/{f}.ses"))

    with open(local_load(f"reference/{f}.txt")) as infile:
        expected = "".join(infile.readlines())
    assert data.summary() == expected
