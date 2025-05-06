"""
Import SESANS data in SasData format
"""

from sasdata.data import SasData
from sasdata.data_util.loader_exceptions import FileContentsException
from sasdata.dataset_types import one_dim
from sasdata.quantities.quantity import Quantity
from sasdata.metadata import Metadata, Sample


def parse_version(lines: list[str]) -> tuple[str, list[str]]:
    import re

    header = lines[0]
    m = re.search("FileFormatVersion\s+(\S+)", header)

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

    return (
        Metadata(
            process=[],
            instrument=None,
            sample=sample,
            title="Title",
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
