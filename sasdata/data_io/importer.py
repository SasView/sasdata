"""
Base class and content-based dispatch for all sasdata importers.

Every module in :mod:`sasdata.data_io.importers` defines a child of :class:`Importer`. Each child knows

* how to recognise a file it can read, from the file *contents* (:meth:`Importer.can_import`), and
* how to read that file into SasMeasurement objects (:meth:`Importer.load_data`).

Children register themselves automatically when they are defined. :meth:`Importer.find_importer` then picks the
appropriate child for any file, so callers do not need to know the file format (or trust the file extension)::

    from sasdata.data_io.importer import Importer

    data = Importer.load("my_file.xml")                 # dict[str, SasMeasurement]
    measurements, errors = Importer().import_data(["a.h5", "b.ses", "c.txt"])
"""

import importlib
import io
from itertools import zip_longest
from pathlib import Path
from typing import IO, ClassVar

from sasdata.data import SasMeasurement

Path_Type = str | Path
Source_Type = Path_Type | IO[bytes]
Value_Type = str | int | float | list[str] | list[int] | list[float]

# Number of bytes read from the start of a file when sniffing its contents.
SNIFF_SIZE = 4096


class Importer:
    """
    Base class for all data importers.

    Child classes must implement :meth:`load_data` and :meth:`can_import`, and may set :attr:`priority` (lower values
    are tested first) and :attr:`name`. Instantiating ``Importer`` directly gives a generic importer whose
    :meth:`load_data` and :meth:`import_data` select the correct child importer for each file automatically.
    """

    # Human-readable name of the format handled by the importer.
    name: ClassVar[str] = "Generic"
    # Order in which the importer is tested when inferring the file type; lower values are tested first.
    # Generic fallback importers (e.g. ASCII) should use a high value.
    priority: ClassVar[int] = 100
    # All concrete importers, populated automatically by :meth:`__init_subclass__`.
    _registry: ClassVar[list[type["Importer"]]] = []

    def __init_subclass__(cls, register: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        if register and cls not in Importer._registry:
            Importer._registry.append(cls)

    def __init__(self, config: dict[str, Value_Type] | None = None):
        self.config: dict[str, Value_Type] = config if config is not None else {}

    # ------------------------------------------------------------------------------------------------------------------
    # Interface to be implemented by every child importer
    # ------------------------------------------------------------------------------------------------------------------

    @classmethod
    def can_import(cls, source: Source_Type) -> bool:
        """Return True if the contents of ``source`` look like a file this importer can read. Must not raise."""
        return False

    def load_data(self, source: Source_Type) -> dict[str, SasMeasurement]:
        """
        Load all measurements contained in ``source``.

        :param source: Path to the file to load (or an open binary file object, where supported).
        :return: A dictionary of the measurements found in the file, keyed by a name unique within the file.
        """
        importer = self.find_importer(source)
        if importer is None:
            raise ValueError(f'Unable to determine the file type of "{source}".')
        return importer(self.config).load_data(source)

    # ------------------------------------------------------------------------------------------------------------------
    # File-type inference
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def importers() -> list[type["Importer"]]:
        """All registered importers, in the order in which they will be tested."""
        # Importing the package imports every importer module, which registers the child classes.
        importlib.import_module("sasdata.data_io.importers")
        return sorted(Importer._registry, key=lambda importer: importer.priority)

    @classmethod
    def find_importer(cls, source: Source_Type) -> type["Importer"] | None:
        """Infer the type of ``source`` from its contents and return the importer class able to read it."""
        for importer in cls.importers():
            if importer.can_import(source):
                return importer
        return None

    @classmethod
    def load(cls, source: Source_Type, config: dict[str, Value_Type] | None = None) -> dict[str, SasMeasurement]:
        """Convenience method: infer the file type of ``source`` and load it."""
        return Importer(config).load_data(source)

    # ------------------------------------------------------------------------------------------------------------------
    # Bulk import
    # ------------------------------------------------------------------------------------------------------------------

    def import_data(self, files: list[Path_Type], configs: list[dict[str, Value_Type]] | None = None) \
            -> tuple[list[SasMeasurement], list[str]]:
        """
        Import a series of files, inferring the importer for each one from its contents.

        :param files: Files to import.
        :param configs: Optional per-file configuration. If there are fewer configs than files, the last config is
            reused for the remaining files.
        :return: All of the measurements loaded, and a list of error messages for anything that could not be loaded.
        """
        output: list[SasMeasurement] = []
        errors: list[str] = []
        if configs is None:
            configs = []
        fill_value = configs[-1] if len(configs) > 0 else None
        for file, config in zip_longest(files, configs, fillvalue=fill_value):
            if file == fill_value:
                # If this is reached, the most likely scenario is len(configs) > len(files) in which case, break out of the loop
                break
            measurements, file_errors = self._import_file(Path(file), config)
            output.extend(measurements)
            errors.extend(file_errors)
        return output, errors

    def _import_file(self, file: Path, config: dict[str, Value_Type] | None = None) \
            -> tuple[list[SasMeasurement], list[str]]:
        if not file.exists():
            return [], [f'File does not exist: "{file}"']
        importer = self.find_importer(file)
        if importer is None:
            return [], [f'Unable to determine the file type of "{file}"']
        merged_config = {**self.config, **(config or {})}
        try:
            imported = list(importer(merged_config).load_data(file).values())
        except Exception as e:
            return [], [f'Error loading "{file}" with the {importer.name} importer: {e}']
        errors = []
        for measurement in imported:
            errors.extend(getattr(measurement, "errors", []))
        return imported, errors


# ----------------------------------------------------------------------------------------------------------------------
# Helpers shared by the importers when sniffing file contents
# ----------------------------------------------------------------------------------------------------------------------

def read_header(source: Source_Type, size: int = SNIFF_SIZE) -> bytes:
    """
    Read up to ``size`` bytes from the start of ``source`` without changing the position of a file object.
    Returns an empty bytes object if the source cannot be read.
    """
    try:
        if isinstance(source, Path_Type):
            path = Path(source)
            if not path.is_file():
                return b""
            with open(path, "rb") as f:
                return f.read(size)
        if isinstance(source, io.IOBase) or hasattr(source, "read"):
            position = source.tell()
            try:
                source.seek(0)
                header = source.read(size)
            finally:
                source.seek(position)
            return header if isinstance(header, bytes) else header.encode()
    except (OSError, ValueError):
        # File access or read errors will fall through here, implying the reader cannot interpret the file.
        # An empty byte string will be returned, but there may be other cases where this try/except are bypassed, so
        # explicitly do nothing here.
        pass
    return b""


def is_text(header: bytes) -> bool:
    """Heuristic check that a file header is text rather than binary data."""
    if not header or b"\x00" in header:
        return False
    try:
        header.decode("utf-8")
    except UnicodeDecodeError as e:
        # The header may have been truncated in the middle of a multibyte character.
        if e.start < len(header) - 4:
            try:
                header.decode("latin-1")
            except UnicodeDecodeError:
                return False
            # latin-1 decodes anything, so reject if there are many control characters.
            controls = sum(1 for b in header if b < 32 and b not in (9, 10, 12, 13))
            return controls / len(header) < 0.05
    return True


def measurements_by_name(measurements: list[SasMeasurement]) -> dict[str, SasMeasurement]:
    """Key a list of measurements by name, adding a numeric suffix to any repeated names so none are lost."""
    output: dict[str, SasMeasurement] = {}
    for measurement in measurements:
        key = measurement.name
        index = 2
        while key in output:
            key = f"{measurement.name} ({index})"
            index += 1
        output[key] = measurement
    return output
