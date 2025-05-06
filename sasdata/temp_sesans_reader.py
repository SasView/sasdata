"""
Import SESANS data in SasData format
"""

from sasdata.data import SasData
from sasdata.data_util.loader_exceptions import FileContentsException
from sasdata.dataset_types import one_dim
from sasdata.quantities.quantity import Quantity
from sasdata.metadata import Metadata, Sample, Instrument, Collimation, Aperture, Vec3
from sasdata.quantities import unit_parser, units
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


def parse_kvs_text(key: str, kvs: dict[str, str]) -> str | None:
    if key not in kvs:
        return None
    return kvs[key]


def parse_sample(kvs: dict[str, str]) -> Sample:
    """Get the sample info from the key value store"""

    thickness = parse_kvs_quantity("Thickness", kvs)
    if thickness is None:
        raise FileContentsException(
            "SES format must include sample thickness to normalise calculations"
        )

    return Sample(
        name=parse_kvs_text("Sample", kvs),
        sample_id=None,
        thickness=thickness,
        transmission=None,
        temperature=None,
        position=None,
        orientation=None,
        details=[],
    )


def parse_instrument(kvs: dict[str, str]) -> Instrument:
    """Get the instrument info from the key value store

    The collimation aperture is used to keep the acceptance angle of the instrument.
    To ensure that this is obviously a virtual aperture, the size is set in kilometers.

    """
    from math import atan

    ymax = parse_kvs_quantity("Theta_ymax", kvs)
    zmax = parse_kvs_quantity("Theta_zmax", kvs)

    if ymax is None:
        raise FileContentsException("SES file must specify Theta_ymax")
    if zmax is None:
        raise FileContentsException("SES file must specify Theta_zmax")

    y : float = atan(ymax.in_units_of(units.radians))
    z : float = atan(ymax.in_units_of(units.radians))

    size = Vec3(
        x=Quantity(0, units.meters),
        y=Quantity(1000*y, units.meters),
        z=Quantity(1000*z, units.meters),
    )

    aperture = Aperture(
        distance=Quantity(value=1, units=units.kilometers),
        size=size,
        size_name=None,
        name="Virtual Acceptance",
        type_="Virtual",
    )
    collimation = Collimation(length=None, apertures=[aperture])
    return Instrument(collimations=[collimation], source=None, detector=[])


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
            instrument=parse_instrument(kvs),
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
