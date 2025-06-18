"""
Unit tests for the new recursive cansas reader
"""

import os
import pytest


from sasdata.temp_sesans_reader import load_data
from sasdata.model_requirements import guess_requirements, ComposeRequirements, SmearModel, SesansModel, NullModel

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

@pytest.mark.sesans
def test_sesans_modelling():
    data = load_data(local_load("sesans_data/sphere_isis.ses"))
    req = guess_requirements(data)
    assert type(guess_requirements(data)) is SesansModel

    ses = SesansModel()
    sme = SmearModel()
    null = NullModel()

    assert type(ses) is SesansModel
    assert type(sme) is SmearModel
    assert type(null) is NullModel

    assert type(ses + sme) is ComposeRequirements
    assert type(null + ses) is SesansModel
    assert type(null + sme) is SmearModel
    assert type(ses + null) is SesansModel
    assert type(sme + null) is SmearModel
