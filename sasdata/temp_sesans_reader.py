"""
Import SESANS data in SasData format
"""

from sasdata.data import SasData
from sasdata.data_util.loader_exceptions import FileContentsException
from sasdata.dataset_types import sesans
from sasdata.quantities.quantity import Quantity
from sasdata.metadata import (
    Metadata,
    Sample,
    MetaNode,
    Process,
)
from sasdata.quantities import unit_parser, units
from collections import defaultdict
from itertools import groupby
import re
import numpy as np


def parse_version(lines: list[str]) -> tuple[str, list[str]]:
    header = lines[0]
    m = re.search(r"FileFormatVersion\s+(\S+)", header)

    if m is None:
        raise FileContentsException(
            "Sesans file does not contain File Format Version header"
        )

    return (m.group(0), lines[1:])


def parse_title(kvs: dict[str, str]) -> str:
    """Get the title from the key value store"""
    if "Title" in kvs:
        return kvs["Title"]
    elif "DataFileTitle" in kvs:
        return kvs["DataFileTitle"]
    for k, v in kvs.items():
        if "Title" in k:
            return v
    return ""


def parse_kvs_quantity(key: str, kvs: dict[str, str]) -> Quantity | None:
    if key not in kvs or key + "_unit" not in kvs:
        return None
    return Quantity(value=float(kvs[key]), units=unit_parser.parse(kvs[key + "_unit"]))


def parse_sample(kvs: dict[str, str]) -> Sample:
    """Get the sample info from the key value store"""

    thickness = parse_kvs_quantity("Thickness", kvs)
    if thickness is None:
        raise FileContentsException(
            "SES format must include sample thickness to normalise calculations"
        )

    return Sample(
        name=kvs.get("Sample"),
        sample_id=None,
        thickness=thickness,
        transmission=None,
        temperature=None,
        position=None,
        orientation=None,
        details=[],
    )


def parse_process(kvs: dict[str, str]) -> Process:
    ymax = parse_kvs_quantity("Theta_ymax", kvs)
    zmax = parse_kvs_quantity("Theta_zmax", kvs)
    orientation = kvs.get("Orientation")

    if ymax is None:
        raise FileContentsException("SES file must specify Theta_ymax")
    if zmax is None:
        raise FileContentsException("SES file must specify Theta_zmax")
    if orientation is None:
        raise FileContentsException("SES file must include encoding orientation")

    terms: dict[str, str | Quantity] = {
        "ymax": ymax,
        "zmax": zmax,
        "orientation": orientation,
    }

    return Process(
        name="SESANS Processing",
        date=None,
        description="Polarisation measurement through a SESANS instrument",
        terms=terms,
        notes=[],
    )


def parse_metanode(kvs: dict[str, str]) -> MetaNode:
    """Convert header into metanode"""
    contents: list[MetaNode] = []
    title = parse_title(kvs)

    for k, v in kvs.items():
        if v.endswith("_unit") and v[:-5] in kvs:
            # This is the unit for another term
            continue
        if v + "_unit" in kvs:
            contents.append(
                MetaNode(
                    name=k,
                    attrs={},
                    contents=Quantity(
                        value=float(v), units=unit_parser.parse(kvs[k + "_unit"])
                    ),
                )
            )
        else:
            contents.append(MetaNode(name=k, attrs={}, contents=v))

    return MetaNode(name=title, attrs={}, contents=contents)


def parse_metadata(lines: list[str]) -> tuple[Metadata, dict[str, str], list[str]]:
    parts = [
        [y for y in x]
        for (_, x) in groupby(lines, lambda x: x.startswith("BEGIN_DATA"))
    ]

    if len(parts) != 3:
        raise FileContentsException("SES file should have exactly one data section")

    # Parse key value store
    kvs: dict[str, str] = {}
    for line in parts[0]:
        m = re.search(r"(\S+)\s+(.+)\n", line)
        if not m:
            continue
        kvs[m.group(1)] = m.group(2)

    return (
        Metadata(
            process=[parse_process(kvs)],
            instrument=None,
            sample=parse_sample(kvs),
            title=parse_title(kvs),
            run=[],
            definition=None,
            raw=parse_metanode(kvs),
        ),
        kvs,
        parts[2],
    )


def parse_data(lines: list[str], kvs: dict[str, str]) -> dict[str, Quantity]:

    data_contents: dict[str, Quantity] = {}
    headers = lines[0].split()
    points = defaultdict(list)
    for line in lines[1:]:
        values = line.split()
        for idx, v in enumerate(values):
            points[headers[idx]].append(float(v))

    for h in points.keys():
        if h.endswith("_error") and h[:-6] in headers:
            # This was an error line
            continue
        unit = units.none
        if h + "_unit" in kvs:
            unit = unit_parser.parse(kvs[h + "_unit"])

        error = None
        if h + "_error" in headers:
            error = np.asarray(points[h + "_error"])

        data_contents[h] = Quantity(
            value=np.asarray(points[h]),
            units=unit,
            standard_error=error,
        )

    for required in ["SpinEchoLength", "Depolarisation", "Wavelength"]:
        if required not in data_contents:
            raise FileContentsException(f"SES file missing {required}")

    return data_contents


def parse_sesans(lines: list[str]) -> SasData:
    version, lines = parse_version(lines)
    metadata, kvs, lines = parse_metadata(lines)
    data_contents = parse_data(lines, kvs)
    return SasData(
        name="Sesans",
        dataset_type=sesans,
        data_contents=data_contents,
        metadata=metadata,
        verbose=False,
    )


def load_data(filename) -> SasData:
    with open(filename) as infile:
        lines = infile.readlines()
    return parse_sesans(lines)
