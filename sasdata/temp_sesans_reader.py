"""
Import SESANS data in SasData format
"""

import re
from itertools import groupby

from sasdata.data import SasData
from sasdata.data_util.loader_exceptions import FileContentsException
from sasdata.dataset_types import one_dim
from sasdata.metadata import (
    Metadata,
    MetaNode,
    Process,
    Sample,
)
from sasdata.quantities import unit_parser
from sasdata.quantities.quantity import Quantity


def parse_version(lines: list[str]) -> tuple[str, list[str]]:
    import re

    header = lines[0]
    m = re.search(r"FileFormatVersion\s+(\S+)", header)

    if m is None:
        raise FileContentsException("Alleged Sesans file does not contain File Format Version header")

    return (m.group(0), lines[1:])

def parse_metadata(lines: list[str]) -> tuple[Metadata, list[str]]:
    sample = Sample(
        name=None,
        sample_id=None,
        thickness=None,
        transmission=None,
        temperature=None,
        position=None,
        orientation=None,
        details=[],
    )


def parse_process(kvs: dict[str, str]) -> Process:
    ymax = parse_kvs_quantity("Theta_ymax", kvs)
    zmax = parse_kvs_quantity("Theta_zmax", kvs)
    orientation = parse_kvs_text("Orientation", kvs)

    if ymax is None:
        raise FileContentsException("SES file must specify Theta_ymax")
    if zmax is None:
        raise FileContentsException("SES file must specify Theta_zmax")
    if orientation is None:
        raise FileContentsException("SES file must include encoding orientation")

    terms: dict[str, str | Quantity[float]] = {
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
        m = re.search("(\\S+)\\s+(.+)\n", line)
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
        ),
        lines,
    )


def parse_data(lines: list[str]) -> dict[str, Quantity]:
    data_contents: dict[str, Quantity] = {}
    return data_contents


def parse_sesans(lines: list[str]) -> SasData:
    version, lines = parse_version(lines)
    metadata, lines = parse_metadata(lines)
    data_contents = parse_data(lines)
    return SasData(
        name="Sesans",
        dataset_type=one_dim,
        data_contents=data_contents,
        metadata=metadata,
        verbose=False,
    )


def load_data(filename) -> SasData:
    with open(filename) as infile:
        lines = infile.readlines()
    return parse_sesans(lines)
