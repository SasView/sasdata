import os
from pathlib import Path

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0.dev"

# An importable path to the example data to
data_path: Path = Path(os.path.join(Path(os.path.dirname(__file__)), 'example_data'))
