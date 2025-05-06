"""
Import SESANS data in SasData format
"""

from sasdata.data import SasData
from sasdata.data_util.loader_exceptions import FileContentsException
from sasdata.dataset_types import one_dim
from sasdata.quantities.quantity import Quantity
from sasdata.metadata import Metadata, Sample
from sasdata.quantities import unit_parser
from itertools import groupby
import re


def parse_version(lines: list[str]) -> tuple[str, list[str]]:
    header = lines[0]
    m = re.search("FileFormatVersion\s+(\S+)", header)

    if m is None:
        raise FileContentsException(
            "Alleged Sesans file does not contain File Format Version header"
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
        name=parse_kvs_quantity("Sample", kvs),
        sample_id=None,
        thickness=thickness,
        transmission=None,
        temperature=None,
        position=None,
        orientation=None,
        details=[],
    )


def parse_metadata(lines: list[str]) -> tuple[Metadata, list[str]]:
    parts = [
        [y for y in x]
        for (_, x) in groupby(lines, lambda x: x.startswith("BEGIN_DATA"))
    ]

    if len(parts) != 3:
        raise FileContentsException("SES file should have exactly one data section")

    # Parse key value store
    kvs: dict[str, str] = {}
    for line in parts[0]:
        m = re.search("(\S+)\s+(.+)\n", line)
        if not m:
            continue
        kvs[m.group(1)] = m.group(2)

    return (
        Metadata(
            process=[],
            instrument=None,
            sample=parse_sample(kvs),
            title=parse_title(kvs),
            run=[],
            definition=None,
        ),
        parts[2],
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
