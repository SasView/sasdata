"""
Base class and dispatch for all sasdata exporters.

Every module in :mod:`sasdata.data_io.exporters` defines a child of :class:`Exporter`. Each child knows how to write a
list of SasMeasurement objects to a single file in one format, in a form that the matching importer in
:mod:`sasdata.data_io.importers` (and SasView) can read back in.

Children register themselves automatically when they are defined. :meth:`Exporter.export_data` selects the exporter
for each file, either by name or from the file extension, checks that the destination can be written to, and then
writes the file::

    from sasdata.data_io.exporter import Exporter

    errors = Exporter().export_data({"out.h5": [data_1, data_2], "out.csv": data_1})
    errors = Exporter().export_data({"no_extension": data_1}, exporter="canSAS XML")
"""

import importlib
import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

from sasdata.data import SasData, SasMeasurement
from sasdata.dataset_types import DatasetType

logger = logging.getLogger(__name__)

Path_Type = str | Path
Value_Type = str | int | float | bool | list[str] | list[int] | list[float]
Data_Type = SasData | Iterable[SasData] | dict[str, SasData]


class ExportError(Exception):
    """Raised when data cannot be exported."""


class Exporter:
    """
    Base class for all data exporters.

    Child classes must implement :meth:`write` and should set :attr:`name`, :attr:`extensions` and
    :attr:`supported_dataset_types`. Instantiating ``Exporter`` directly gives a generic exporter whose
    :meth:`export_data` selects the correct child exporter for each file.
    """

    # Human-readable name of the format written by the exporter. Used to select the exporter by name.
    name: ClassVar[str] = "Generic"
    # Alternative names that can be used to select the exporter.
    aliases: ClassVar[tuple[str, ...]] = ()
    # File extensions (lower case, including the leading '.') associated with the format. The first is the default.
    extensions: ClassVar[tuple[str, ...]] = ()
    # Dataset types that can be written to, and read back from, the format. An empty tuple means any type.
    supported_dataset_types: ClassVar[tuple[DatasetType, ...]] = ()
    # Whether more than one dataset can be written to a single file.
    multiple_datasets: ClassVar[bool] = True
    # All concrete exporters, populated automatically by :meth:`__init_subclass__`.
    _registry: ClassVar[list[type["Exporter"]]] = []

    def __init_subclass__(cls, register: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        if register and cls not in Exporter._registry:
            Exporter._registry.append(cls)

    def __init__(self, config: dict[str, Value_Type] | None = None):
        self.config: dict[str, Value_Type] = config if config is not None else {}

    # ------------------------------------------------------------------------------------------------------------------
    # Interface to be implemented by every child exporter
    # ------------------------------------------------------------------------------------------------------------------

    def write(self, data: dict[str, SasMeasurement], path: Path):
        """
        Write ``data`` to ``path``. The data and destination have already been validated when this is called.

        :param data: The measurements to write, keyed by a unique name. Formats that can hold several datasets use the
            keys to name them, so the same keys are returned when the file is imported.
        :param path: The file to write. This is a temporary file in the destination directory, which is moved to the
            final destination once it has been written successfully.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement write()")

    # ------------------------------------------------------------------------------------------------------------------
    # Exporter selection
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def exporters() -> list[type["Exporter"]]:
        """All registered exporters."""
        # Importing the package imports every exporter module, which registers the child classes.
        importlib.import_module("sasdata.data_io.exporters")
        return list(Exporter._registry)

    @classmethod
    def find_exporter(cls, path: Path_Type | None = None, name: str | type["Exporter"] | None = None) \
            -> type["Exporter"] | None:
        """
        Select an exporter by name (case-insensitive, checking :attr:`name` and :attr:`aliases`), or if no name is
        given, from the extension of ``path``.
        """
        exporters = cls.exporters()
        if isinstance(name, type):
            return name if issubclass(name, Exporter) and name in exporters else None
        if name is not None:
            wanted = name.strip().lower()
            for exporter in exporters:
                if wanted in (n.lower() for n in (exporter.name, *exporter.aliases)):
                    return exporter
            return None
        if path is not None:
            extension = Path(path).suffix.lower()
            for exporter in exporters:
                if extension in exporter.extensions:
                    return exporter
        return None

    # ------------------------------------------------------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def validate_destination(path: Path_Type, overwrite: bool = True) -> Path:
        """
        Check that ``path`` can be written to: its directory must exist, be a directory, and be writable, and if the
        file already exists it must be a writable file and ``overwrite`` must be True.

        :return: The absolute path to the file.
        :raises FileNotFoundError, NotADirectoryError, IsADirectoryError, FileExistsError, PermissionError:
        """
        file = Path(path).expanduser().absolute()
        directory = file.parent
        if not directory.exists():
            raise FileNotFoundError(f'The directory "{directory}" does not exist.')
        if not directory.is_dir():
            raise NotADirectoryError(f'"{directory}" is not a directory.')
        if file.is_dir():
            raise IsADirectoryError(f'"{file}" is a directory.')
        if not os.access(directory, os.W_OK | os.X_OK):
            raise PermissionError(f'You do not have permission to write to the directory "{directory}".')
        if file.exists():
            if not overwrite:
                raise FileExistsError(f'"{file}" already exists.')
            if not os.access(file, os.W_OK):
                raise PermissionError(f'You do not have permission to overwrite "{file}".')
        # os.access does not reflect every permission system (e.g. Windows ACLs), so confirm by creating a file.
        try:
            handle, probe = tempfile.mkstemp(dir=directory, prefix=".sasdata_", suffix=".tmp")
        except OSError as e:
            raise PermissionError(f'You do not have permission to write to the directory "{directory}": {e}') from e
        os.close(handle)
        os.remove(probe)
        return file

    def validate_data(self, data: list[SasData]):
        """
        Check the data can be written by this exporter.

        :raises ExportError: If there is no data, too many datasets, or a dataset type the format cannot hold.
        """
        if len(data) == 0:
            raise ExportError("There is no data to export.")
        if len(data) > 1 and not self.multiple_datasets:
            raise ExportError(f"The {self.name} format can only hold one dataset per file, but {len(data)} were given.")
        for datum in data:
            if not isinstance(datum, SasMeasurement):
                raise ExportError(f'"{datum.name}" is a {type(datum).__name__}, which cannot be exported yet.')
            if self.supported_dataset_types and datum.dataset_type not in self.supported_dataset_types:
                supported = ", ".join(t.name for t in self.supported_dataset_types)
                raise ExportError(f'"{datum.name}" is {datum.dataset_type.name} data, but the {self.name} format '
                                  f'only supports: {supported}.')

    # ------------------------------------------------------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------------------------------------------------------

    def export_data(self, file_map: dict[Path_Type, Data_Type], exporter: str | type["Exporter"] | None = None,
                    overwrite: bool = True) -> list[str]:
        """
        Export data to one or more files.

        :param file_map: Map of each destination file to the data to write to it. The data may be a single dataset,
            a list of datasets, or a dictionary of datasets as returned by the importers.
        :param exporter: The name (or class) of the exporter to use for every file. If None and this is a child
            exporter, that exporter is used; otherwise, the exporter is chosen from each file extension.
        :param overwrite: Whether existing files may be replaced.
        :return: A list of error messages for anything that could not be exported. An empty list means success.
        """
        errors: list[str] = []
        for path, data in file_map.items():
            names = "data"
            try:
                data_map = keyed_data(data)
                names = ", ".join(data_map) or names
                exporter_class = self._select_exporter(path, exporter)
                file = self.validate_destination(path, overwrite)
                instance = self if type(self) is exporter_class else exporter_class(self.config)
                instance.validate_data(list(data_map.values()))
                instance._write_safely(data_map, file)
            except Exception as e:
                logger.debug("Export failed", exc_info=True)
                errors.append(f'Unable to export {names} to "{path}": {e}')
        return errors

    @classmethod
    def export(cls, data: Data_Type, path: Path_Type, exporter: str | type["Exporter"] | None = None,
               config: dict[str, Value_Type] | None = None, overwrite: bool = True):
        """
        Convenience method: export ``data`` to a single file.

        :raises ExportError: If the data could not be exported.
        """
        instance = cls(config)
        errors = instance.export_data({path: data}, exporter=exporter, overwrite=overwrite)
        if errors:
            raise ExportError("\n".join(errors))

    def _select_exporter(self, path: Path_Type, exporter: str | type["Exporter"] | None) -> type["Exporter"]:
        if exporter is not None:
            selected = self.find_exporter(name=exporter)
            if selected is None:
                known = ", ".join(e.name for e in self.exporters())
                raise ExportError(f'Unknown exporter "{exporter}". Known exporters are: {known}.')
            return selected
        if type(self) is not Exporter:
            return type(self)
        selected = self.find_exporter(path=path)
        if selected is None:
            raise ExportError(f'Unable to determine the export format from the extension of "{path}". '
                              f'Choose an exporter or use one of: {", ".join(all_extensions())}.')
        return selected

    def _write_safely(self, data: dict[str, SasMeasurement], file: Path):
        """
        Write to a temporary file in the destination directory then move it into place, so an existing file is never
        left half-written if the export fails.
        """
        handle, temp_name = tempfile.mkstemp(dir=file.parent, prefix=f".{file.name}.", suffix=".tmp")
        os.close(handle)
        temp_file = Path(temp_name)
        try:
            self.write(data, temp_file)
            # mkstemp creates files readable only by the owner; give the export the permissions a new file would have
            # (or keep those of the file being replaced).
            os.chmod(temp_file, file.stat().st_mode if file.exists() else default_file_mode())
            os.replace(temp_file, file)
        finally:
            if temp_file.exists():
                temp_file.unlink()


# ----------------------------------------------------------------------------------------------------------------------
# Helpers shared by the exporters
# ----------------------------------------------------------------------------------------------------------------------

def keyed_data(data: Data_Type) -> dict[str, SasData]:
    """
    Convert a single dataset, an iterable of datasets or a dict of datasets into a dict of datasets with unique keys.
    The keys of a dict are kept; otherwise, the datasets are keyed by name.
    """
    if isinstance(data, dict):
        return {str(key): value for key, value in data.items()}
    data_list = [data] if isinstance(data, SasData) else list(data)
    return dict(zip(unique_names(data_list), data_list))


def default_file_mode() -> int:
    """The permissions given to a newly created file, taking the umask into account."""
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def all_extensions() -> list[str]:
    """Every file extension that an exporter is registered for."""
    return [extension for exporter in Exporter.exporters() for extension in exporter.extensions]


def unique_names(data: list[SasData]) -> list[str]:
    """The names of the datasets, with a numeric suffix added to any repeated names."""
    names: list[str] = []
    for index, datum in enumerate(data):
        name = str(getattr(datum, "name", "") or f"SasData{index + 1:02d}")
        candidate = name
        suffix = 2
        while candidate in names:
            candidate = f"{name} ({suffix})"
            suffix += 1
        names.append(candidate)
    return names
