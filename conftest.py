import pytest

from sasdata.metadata import Instrument, Metadata, Source
from sasdata.quantities.quantity import Quantity
from sasdata.quantities.units import angstroms


def pytest_addoption(parser):
    parser.addoption("--show_plots", action="store_true", default=False, help="Display diagnostic plots during tests")


@pytest.fixture
def show_plots(request):
    return request.config.getoption("--show_plots")


@pytest.fixture
def basic_metadata():
    """A minimal metadata object."""
    wavelength = Quantity(1.0, angstroms)
    source = Source(
        radiation=None,
        beam_shape=None,
        beam_size=None,
        wavelength=wavelength,
        wavelength_max=None,
        wavelength_min=None,
        wavelength_spread=None,
    )
    instrument = Instrument(collimations=[], source=source, detector=[])
    metadata = Metadata(title=None, run=[], definition=None, process=[], sample=None, instrument=instrument, raw=None)

    return metadata
