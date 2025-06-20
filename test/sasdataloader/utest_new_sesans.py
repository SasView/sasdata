"""
Unit tests for the new recursive cansas reader
"""

import numpy as np
import os
import pytest


from sasdata.model_requirements import guess_requirements, ComposeRequirements, SmearModel, SesansModel, NullModel
from sasdata.temp_sesans_reader import load_data
from sasdata.quantities import units, unit_parser

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
    import matplotlib.pyplot as plt
    data = load_data(local_load("sesans_data/sphere2micron.ses"))
    req = guess_requirements(data)
    assert type(req) is SesansModel


    def sphere(qr):
        def sas_3j1x_x(x):
            return (np.sin(x) - x * np.cos(x))/x**3
        def form_volume(x):
            return np.pi * 4.0 / 3.0 * x**3
        radius = 10000 # 1 micron

        bes = sas_3j1x_x(q*radius)
        contrast = 5.4e-7 # Contrast is hard coded for best fit
        form = contrast * form_volume(radius) * bes
        f2 = 1.0e-4*form*form
        return f2

    # The Hankel transform of x is -r^-3
    x = data._data_contents["SpinEchoLength"].in_units_of(units.angstroms)
    q = req.preprocess_q(x, data)
    result = req.postprocess_iq(sphere(q), data)

    y, yerr = data._data_contents["Depolarisation"].in_units_of_with_standard_error(unit_parser.parse("A-2 cm-1"))
    assert y.shape == result.shape

    xi_squared = np.sum( ((y - result) / yerr)**2 ) / len(y)
    assert 1.0 < xi_squared < 1.5



@pytest.mark.sesans
def test_model_algebra():
    ses = SesansModel()
    sme = SmearModel()
    null = NullModel()

    assert type(ses) is SesansModel
    assert type(sme) is SmearModel
    assert type(null) is NullModel

    # Ignore slit smearing if we perform a sesans transform afterwards
    assert type(sme + ses) is SesansModel
    # However, it is possible for the spin echo lengths to have some
    # smearing between them.
    assert type(ses + sme) is ComposeRequirements
    assert type(null + ses) is SesansModel
    assert type(null + sme) is SmearModel
    assert type(ses + null) is SesansModel
    assert type(sme + null) is SmearModel
