from pathlib import Path

from sasdata.data import SasData

Path_Type = str | Path


class Import:
    def __init__(self):
        pass

    def import_data(self, files: Path_Type) -> (list[SasData], list[str]):
        output = []
        errors = []
        for file in files:
            file = Path(file)
            if file.exists():
                pass
            else:
                errors.append(f'File "{file}" not found.')
        return output, errors
