"""
Configuration for the importers: where in a file each piece of metadata is found.

The importers read the metadata that the file format defines (NXcanSAS, canSAS XML and SESANS headers). Many files
store metadata somewhere else, or not at all in the standard place. An :class:`ImporterConfig` tells any importer where
to find the value of named metadata fields, and the importer uses that value in place of the one it read from the
standard location. The configuration has the same form as the read configuration of SasModelsExplorer: a YAML document
(or the equivalent dict) with a ``pathDict`` mapping names to locations in the file::

    pathDict:
      title: /entry/title                           # HDF5: absolute path
      sample.name: sample/name                      # HDF5: relative to each SASentry group
      sample.thickness: sample/thickness            # units from the dataset's "units" attribute
      sample.temperature: "sample/temperature@value"
      instrument.source.wavelength: beam/wavelength
    units:
      sample.thickness: mm                          # used instead of any units found in the file

The configuration is passed to any importer, and is also accepted by :meth:`Importer.load`::

    config = ImporterConfig.from_file("my_config.yaml")
    data = Importer.load("my_file.h5", config)
    data = Importer.load("my_file.h5", {"pathDict": {"title": "/entry/title"}})

Metadata names
--------------
The names in ``pathDict`` and ``units`` are the dotted names in :data:`METADATA_FIELDS`, which follow the layout of
:class:`sasdata.metadata.Metadata`. Where the metadata holds a list of objects (the detectors, collimations and
processes), the name refers to the first one, which is created if the file does not have one.

Locations
---------
The syntax of a location depends on the kind of file, and is shared by the formats with a tree of named nodes.

HDF5 and XML files
    A ``/`` separated path, optionally followed by ``@attribute`` to read an attribute of the node instead of its
    value. A path starting with ``/`` is absolute. Any other path is relative to the ``SASentry`` of each measurement
    in the file. XML elements are matched by their local names, so the namespace of the file does not matter, and
    ``name[2]`` picks the second element called ``name``. ``*`` matches any name.

Text files (ASCII and SESANS), searching the lines above the data
    ``line:N`` is the whole of line ``N``, counting from 1. ``header:Key`` is the value that follows ``Key:``,
    ``Key=`` or ``Key `` at the start of a line, ignoring case and any comment character. ``regex:Pattern`` is the
    first group of a regular expression that is searched for in each line, or the whole match if it has no groups.

A location that is not found is skipped with a warning, so one configuration can be used for files that do not all
have every item. A value that cannot be converted raises :class:`ImporterConfigError`.

Units
-----
A quantity takes its units from the ``units`` section of the configuration, else from the node's ``units`` (or
``unit``) attribute, else from the text of the value itself, as in ``1.5 mm``.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from lxml import etree

from sasdata.metadata import Collimation, Detector, Instrument, Metadata, Process, Sample, Source
from sasdata.quantities import unit_parser
from sasdata.quantities.quantity import Quantity

logger = logging.getLogger(__name__)

# The metadata that can be located. The value is how the text found at the location is converted:
#   str       text
#   float     a number
#   quantity  a number with units
#   list      a list of text values
METADATA_FIELDS: dict[str, str] = {
    "title": "str",
    "run": "list",
    "definition": "str",
    "sample.name": "str",
    "sample.sample_id": "str",
    "sample.transmission": "float",
    "sample.thickness": "quantity",
    "sample.temperature": "quantity",
    "sample.details": "list",
    "instrument.source.radiation": "str",
    "instrument.source.beam_shape": "str",
    "instrument.source.wavelength": "quantity",
    "instrument.source.wavelength_min": "quantity",
    "instrument.source.wavelength_max": "quantity",
    "instrument.source.wavelength_spread": "quantity",
    "instrument.detector.name": "str",
    "instrument.detector.distance": "quantity",
    "instrument.detector.slit_length": "quantity",
    "instrument.collimation.length": "quantity",
    "process.name": "str",
    "process.date": "str",
    "process.description": "str",
    "process.notes": "list",
}

# Dataclass fields that hold lists, which are empty rather than None when a new object is made.
_LIST_FIELDS = {"details", "apertures", "notes", "collimations", "detector", "run", "process"}
_DICT_FIELDS = {"terms"}

_NUMBER_WITH_UNITS = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")


class ImporterConfigError(ValueError):
    """The configuration is invalid, or a value it locates cannot be converted."""


@dataclass
class Located:
    """A value found in a file, and the attributes of the node it was found in (which may name its units)."""

    value: Any
    attrs: dict[str, str] = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class ImporterConfig:
    """Where to find metadata in a file, and the units of what is found. See the module documentation."""

    # Location in the file of each named item of metadata
    pathDict: dict[str, str] = field(default_factory=dict)
    # Units of each named item, used in place of any units found in the file
    units: dict[str, str] = field(default_factory=dict)
    # Anything else in the configuration, kept for the importers that use it
    extra: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.pathDict)

    # -- Construction -------------------------------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ImporterConfig:
        """Check and normalise a configuration given as a dictionary."""
        raw = dict(raw)
        path_dict = _as_str_mapping(raw.pop("pathDict", None), "pathDict")
        unit_dict = _as_str_mapping(raw.pop("units", None), "units")
        for section, mapping in (("pathDict", path_dict), ("units", unit_dict)):
            unknown = [key for key in mapping if key not in METADATA_FIELDS]
            if unknown:
                raise ImporterConfigError(
                    f"Unknown metadata name(s) in {section}: {', '.join(unknown)}. "
                    f"The names that can be used are: {', '.join(METADATA_FIELDS)}"
                )
        empty = [key for key, location in path_dict.items() if not location.strip()]
        if empty:
            raise ImporterConfigError(f"The location of {', '.join(empty)} in pathDict is empty")
        for key, unit in unit_dict.items():
            try:
                unit_parser.parse(unit)
            except Exception as e:
                raise ImporterConfigError(f'Cannot read the units "{unit}" given for {key}: {e}') from e
        return cls(pathDict=path_dict, units=unit_dict, extra=raw)

    @classmethod
    def from_yaml(cls, text: str) -> ImporterConfig:
        """Read a configuration from YAML text. Empty text gives an empty configuration."""
        try:
            raw = yaml.safe_load(text) if text.strip() else {}
        except yaml.YAMLError as e:
            raise ImporterConfigError(f"YAML error: {e}") from e
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ImporterConfigError("The configuration must be a YAML mapping")
        return cls.from_dict(raw)

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> ImporterConfig:
        """Read a configuration from a YAML file."""
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def coerce(cls, config: ConfigLike) -> ImporterConfig:
        """
        Accept any of the ways of giving a configuration: None, an ImporterConfig, a dictionary, a Path to a YAML file,
        or a string, which is the path of a YAML file if one exists with that name and is otherwise YAML text.
        """
        match config:
            case None:
                return cls()
            case ImporterConfig():
                return config
            case Mapping():
                return cls.from_dict(config)
            case Path():
                return cls.from_file(config)
            case str():
                if "\n" not in config and os.path.isfile(config):
                    return cls.from_file(config)
                return cls.from_yaml(config)
        raise TypeError(f"Cannot make an importer configuration from a {type(config).__name__}")

    def merged(self, other: ImporterConfig | None) -> ImporterConfig:
        """A configuration with everything in this one, overridden by anything given in ``other``."""
        if other is None:
            return self
        return replace(
            self,
            pathDict={**self.pathDict, **other.pathDict},
            units={**self.units, **other.units},
            extra={**self.extra, **other.extra},
        )

    # -- Use ----------------------------------------------------------------------------------------------------------

    def apply_to_metadata(self, metadata: Metadata, source: MetadataSource) -> Metadata:
        """
        Set each item of metadata given in ``pathDict`` from its location in ``source``, replacing what the importer
        read from the standard location. ``metadata`` is changed in place, and returned.
        """
        for key, location in self.pathDict.items():
            matches = source.read(location)
            if not matches:
                logger.warning(f'The location "{location}" configured for {key} was not found')
                continue
            value = self._convert(key, matches)
            _assign(metadata, key.split("."), value)
        return metadata

    def _convert(self, key: str, matches: list[Located]) -> Any:
        kind = METADATA_FIELDS[key]
        first = matches[0]
        try:
            match kind:
                case "str":
                    return str(first.value).strip()
                case "list":
                    return [str(match.value).strip() for match in matches]
                case "float":
                    return float(first.value)
                case "quantity":
                    return self._quantity(key, first)
        except ImporterConfigError:
            raise
        except (TypeError, ValueError) as e:
            raise ImporterConfigError(f'Cannot convert the value "{first.value}" found for {key}: {e}') from e
        raise AssertionError(kind)

    def _quantity(self, key: str, located: Located) -> Quantity:
        value, unit_text = located.value, None
        if isinstance(value, (str, bytes)):
            text = value.decode() if isinstance(value, bytes) else value
            parsed = _NUMBER_WITH_UNITS.match(text)
            if parsed is None:
                raise ValueError("not a number")
            value, unit_text = float(parsed.group(1)), parsed.group(2) or None
        # Units given in the configuration take precedence over those in the file
        unit_text = self.units.get(key) or located.attrs.get("units") or located.attrs.get("unit") or unit_text
        if not unit_text:
            raise ImporterConfigError(
                f"The units of {key} are not in the file, so they must be given in the units section of the "
                "configuration"
            )
        try:
            unit = unit_parser.parse(str(unit_text).strip())
        except Exception as e:
            raise ImporterConfigError(f'Cannot read the units "{unit_text}" of {key}: {e}') from e
        return Quantity(float(value), unit)


ConfigLike = ImporterConfig | Mapping[str, Any] | str | os.PathLike | None


def _as_str_mapping(value: Any, name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ImporterConfigError(f"{name} must be a mapping")
    return {str(key): str(item) for key, item in value.items()}


# ----------------------------------------------------------------------------------------------------------------------
# Setting metadata
# ----------------------------------------------------------------------------------------------------------------------

def _empty(cls):
    """An object of a metadata class with nothing set"""
    values = {}
    for f in fields(cls):
        if f.name in _LIST_FIELDS:
            values[f.name] = []
        elif f.name in _DICT_FIELDS:
            values[f.name] = {}
        else:
            values[f.name] = None
    return cls(**values)


def _ensure(obj, attribute: str, cls):
    """The object held in an attribute, which is created if there isn't one"""
    value = getattr(obj, attribute)
    if value is None:
        value = _empty(cls)
        setattr(obj, attribute, value)
    return value


def _first(obj, attribute: str, cls):
    """The first object in a list held in an attribute, which is created if there isn't one"""
    items = getattr(obj, attribute)
    if items is None:
        items = []
        setattr(obj, attribute, items)
    if not items:
        items.append(_empty(cls))
    return items[0]


def _assign(metadata: Metadata, parts: list[str], value: Any) -> None:
    """Set the item of metadata named by the parts of a dotted name"""
    match parts[:-1]:
        case []:
            target = metadata
        case ["sample"]:
            target = _ensure(metadata, "sample", Sample)
        case ["instrument", "source"]:
            target = _ensure(_ensure(metadata, "instrument", Instrument), "source", Source)
        case ["instrument", "detector"]:
            target = _first(_ensure(metadata, "instrument", Instrument), "detector", Detector)
        case ["instrument", "collimation"]:
            target = _first(_ensure(metadata, "instrument", Instrument), "collimations", Collimation)
        case ["process"]:
            target = _first(metadata, "process", Process)
        case _:
            raise ImporterConfigError(f"Cannot set {'.'.join(parts)}")
    setattr(target, parts[-1], value)


# ----------------------------------------------------------------------------------------------------------------------
# Sources of metadata: each reads the values at a location in one kind of file
# ----------------------------------------------------------------------------------------------------------------------

class MetadataSource:
    """The contents of a file, or part of one, in which the locations of a configuration are found."""

    def read(self, location: str) -> list[Located]:
        """All of the values found at a location, or an empty list if there are none."""
        raise NotImplementedError


def _text(value: Any) -> Any:
    """Plain Python text or numbers in place of the bytes and NumPy types that files hold"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


class Hdf5Source(MetadataSource):
    """Locations in an HDF5 file: ``/absolute/path`` or ``path/relative/to/the/entry``, then optionally ``@attribute``."""

    def __init__(self, entry):
        self.entry = entry

    def read(self, location: str) -> list[Located]:
        path, _, attribute = location.strip().partition("@")
        base = self.entry.file if path.startswith("/") else self.entry
        if path.strip("/") == "":
            node = base if path.startswith("/") else self.entry
        elif path in base:
            node = base[path]
        else:
            return []
        if attribute:
            if attribute not in node.attrs:
                return []
            return [Located(_text(v)) for v in np.atleast_1d(node.attrs[attribute]).ravel()]
        if not hasattr(node, "dtype"):
            logger.warning(f'The location "{location}" is a group, not a value')
            return []
        attrs = {name: str(_text(np.atleast_1d(v).ravel()[0])) for name, v in node.attrs.items() if np.size(v) > 0}
        data = node.asstr()[()] if node.dtype.kind in "SO" else node[()]
        return [Located(_text(v), attrs) for v in np.atleast_1d(data).ravel()]


class XmlSource(MetadataSource):
    """Locations in an XML file: ``/absolute/path`` or ``path/relative/to/the/entry``, then optionally ``@attribute``."""

    _SEGMENT = re.compile(r"^(?P<tag>[^\[\]]+?)(?:\[(?P<index>\d+)\])?$")

    def __init__(self, entry: etree._Element):
        self.entry = entry

    def read(self, location: str) -> list[Located]:
        path, _, attribute = location.strip().partition("@")
        segments = [s for s in path.split("/") if s]
        if path.startswith("/"):
            root = self.entry.getroottree().getroot()
            if not segments or not self._matches(root, segments[0]):
                return []
            candidates, segments = [root], segments[1:]
        else:
            candidates = [self.entry]
        for segment in segments:
            candidates = [child for node in candidates for child in self._children(node, segment)]
        if attribute:
            return [Located(node.attrib[attribute]) for node in candidates if attribute in node.attrib]
        return [Located("".join(node.itertext()).strip(), dict(node.attrib)) for node in candidates]

    def _parse_segment(self, segment: str) -> tuple[str, int | None]:
        parsed = self._SEGMENT.match(segment)
        if parsed is None:
            raise ImporterConfigError(f'Invalid XML location segment "{segment}"')
        index = parsed.group("index")
        return parsed.group("tag"), int(index) if index else None

    def _matches(self, node: etree._Element, segment: str) -> bool:
        tag, _ = self._parse_segment(segment)
        return isinstance(node.tag, str) and tag in ("*", etree.QName(node).localname)

    def _children(self, node: etree._Element, segment: str) -> list[etree._Element]:
        _, index = self._parse_segment(segment)
        children = [child for child in node if self._matches(child, segment)]
        if index is None:
            return children
        return children[index - 1:index]


class TextSource(MetadataSource):
    """Locations in the header lines of a text file: ``line:N``, ``header:Key`` or ``regex:Pattern``."""

    def __init__(self, lines: list[str]):
        self.lines = [line.rstrip("\r\n") for line in lines]

    def read(self, location: str) -> list[Located]:
        kind, _, argument = location.strip().partition(":")
        match kind.strip().lower():
            case "line":
                try:
                    number = int(argument)
                except ValueError as e:
                    raise ImporterConfigError(f'Invalid line number in "{location}"') from e
                return [Located(self.lines[number - 1].strip())] if 1 <= number <= len(self.lines) else []
            case "header":
                key = re.escape(argument.strip())
                pattern = re.compile(rf"^\s*[#;%]*\s*{key}\s*(?:[:=]|\s)\s*(.*?)\s*$", re.IGNORECASE)
            case "regex":
                try:
                    pattern = re.compile(argument)
                except re.error as e:
                    raise ImporterConfigError(f'Invalid regular expression in "{location}": {e}') from e
            case _:
                raise ImporterConfigError(
                    f'"{location}" is not a location in a text file. Use line:N, header:Key or regex:Pattern'
                )
        found = []
        for line in self.lines:
            match = pattern.search(line)
            if match:
                found.append(Located((match.group(1) if pattern.groups else match.group(0)).strip()))
        return found
