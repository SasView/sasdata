import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--show_plots", action="store_true", default=False, help="Display diagnostic plots during tests"
    )

@pytest.fixture
def show_plots(request):
    return request.config.getoption("--show_plots")
