import mimetypes
from itertools import zip_longest
from pathlib import Path

from sasdata.data import SasMeasurement
from sasdata.data_io.importers import *

Path_Type = str | Path
Value_Type = str | int | float | list[str] | list[int] | list[float]


class Importer:
    def __init__(self):
        pass

    def import_data(self, files: list[Path_Type], configs: list[dict[str, Value_Type]] | None = None) \
            -> (list[SasMeasurement], list[str]):
        output = []
        errors = []
        if configs is None:
            configs = []
        fill_value = configs[-1] if len(configs) > 0 else None
        for file, config in zip_longest(files, configs, fillvalue=fill_value):
            file = Path(file)
            measurements, errors = self._import_from_url(file, config)
            output.extend(measurements)
            errors.extend(errors)
        return output, errors

    def _import_from_url(self, url: Path_Type, config: dict[str, Value_Type] | None = None) -> (list[SasMeasurement], list[str]):
        file = Path(url)
        if not file.exists():
            return [], [f'File does not exist: "{file}"']
        # Deprecation Warning:
        #     mimetypes.guess_type() is soft deprecated in favor of mimetypes.guess_file_type() as of python 3.13
        #     Once we move forward from python 3.12, this line should change
        mime_type, encoding = mimetypes.guess_type(url)
        imported = []
        errors = []
        # TODO: Need a viable config system that can pass a dictionary of value locations to each importer
        try:
            match mime_type:
                case 'application/xml' | 'text/xml':
                    file_dict = import_xml.load_data(file)
                    imported.extend(list(file_dict.values()))
                case 'application/vnd.hdfgroup.hdf5':
                    file_list = import_hdf5.load_data(file)
                    imported.extend(file_list)
                case _:
                    file_list = import_ascii.load_data_default_params(file)
                    imported.extend(file_list)
        except Exception as e:
            errors.append(f'Error accessing "{file}": {e}')
        for file in imported:
            if hasattr(file, "errors"):
                errors.extend(file.errors)
        return imported, errors
