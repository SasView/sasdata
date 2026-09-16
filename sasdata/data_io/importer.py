from pathlib import Path

from sasdata.data import SasMeasurement
from sasdata.data_io.importers import *

Path_Type = str | Path


class Importer:
    def __init__(self):
        pass

    def import_data(self, files: Path_Type) -> (list[SasMeasurement], list[str]):
        output = []
        errors = []
        for file in files:
            file = Path(file)
            if measurements := self._import_from_url(file):
                output.append(measurements)
            else:
                errors.append(f'File "{file}" not found.')
        return output, errors

    def _import_from_url(self, url: Path_Type) -> list[SasMeasurement]:
        file = Path(url)
        if not file.exists():
            return []
        with open(url) as f:
            pass
        return []
