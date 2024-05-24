import os
from pathlib import Path

__version__ = "0.9.0"

# An importable path to the example data to
data_path: Path = Path(os.path.join(Path(os.path.dirname(__file__)), 'example_data'))
