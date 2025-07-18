"""
Unit tests for the new recursive cansas reader
"""

import numpy as np
import os
import pytest


from sasdata.model_requirements import guess_requirements, ComposeRequirements, PinholeModel, SesansModel, NullModel, ModellingRequirements
from sasdata.temp_sesans_reader import load_data
from sasdata.quantities.quantity import Quantity
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
    data = load_data(local_load("sesans_data/sphere2micron.ses"))
    req = guess_requirements(data)
    assert type(req) is SesansModel

    def form_volume(x):
        return np.pi * 4.0 / 3.0 * x**3

    radius = 10000
    contrast = 5.4e-7 # Contrast is hard coded for best fit
    form = contrast * form_volume(radius)
    f2 = 1.0e-4*form*form

    xi_squared = smear(req, data._data_contents["SpinEchoLength"].value, full_data=data, y = data._data_contents["Depolarisation"].in_units_of(unit_parser.parse("A-2 cm-1")) / f2, radius=radius)
    assert 1.0 < xi_squared < 1.5

@pytest.mark.sesans
def test_pinhole_zero():
    assert pinhole_smear(0) == 0

@pytest.mark.sesans
def test_pinhole_smear():
    smearing = [10**x for x in range(-3, 3)]
    smears = [pinhole_smear(x) for x in smearing]
    old = 0
    for factor, smear in zip(smearing, smears):
        print(factor, smear)
        assert old < smear < 1.2
        old = smear
    assert pinhole_smear(3**6) > 1.2


def pinhole_smear(smearing: float):
    data = Quantity(np.linspace(1e-4, 1e-1, 1000), units.per_angstrom)
    req = PinholeModel(np.diff(data.value, prepend=0) * smearing)
    return smear(req, data.value)


def smear(req: ModellingRequirements, data: np.ndarray, y=None, full_data=None, radius=200):
    def sphere(q):
        def sas_3j1x_x(x):
            return (np.sin(x) - x * np.cos(x))/x**3

        return sas_3j1x_x(q * radius)**2

    inner_q = req.preprocess_q(data, full_data)
    result = req.postprocess_iq(sphere(inner_q), data)

    if y is None:
        y = sphere(data)

    xi_squared = np.sum( ((y - result)/result )**2 ) / len(y)
    return xi_squared



@pytest.mark.sesans
def test_model_algebra():
    ses = SesansModel()
    pin = PinholeModel(np.linspace(1e-3, 1, 1000))
    null = NullModel()

    # Ignore slit smearing if we perform a sesans transform afterwards
    assert type(pin + ses) is SesansModel
    # However, it is possible for the spin echo lengths to have some
    # smearing between them.
    assert type(ses + pin) is ComposeRequirements
    assert type(null + ses) is SesansModel
    assert type(null + pin) is PinholeModel
    assert type(ses + null) is SesansModel
    assert type(pin + null) is PinholeModel
