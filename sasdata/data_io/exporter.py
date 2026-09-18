from pathlib import Path

from sasdata.data import SasData

Path_Type = str | Path


class Exporter:
    def __init__(self):
        pass

    def export_data(self, file_map: dict[Path_Type, list[SasData]]) -> list[str]:
        errors = []
        for path, sas_data_list in file_map:
            file = Path(path)
            try:
                with open(file, 'w') as export_file:
                    pass
            except Exception:
                errors.append(f'Unable to write {', '.join([data.name for data in sas_data_list])} to "{file}".')
        return errors
