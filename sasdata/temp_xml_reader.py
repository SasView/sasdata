import logging
from lxml import etree
import numpy as np
from typing import Callable

from sasdata.data import SasData
from sasdata.dataset_types import one_dim
from sasdata.metadata import (
    Instrument,
    Collimation,
    Aperture,
    Source,
    BeamSize,
    Detector,
    Vec3,
    Rot3,
    Sample,
    Process,
    Metadata,
)
from sasdata.quantities.quantity import Quantity
import sasdata.quantities.unit_parser as unit_parser
from sasdata.quantities.units import Unit, none as unitless

logger = logging.getLogger(__name__)

test_file = "./example_data/1d_data/ISIS_Polymer_Blend_TK49.xml"

ns = {
    "cansas11": "urn:cansas1d:1.1",
    "cansas10": "urn:cansas1d:1.0",
    "cansas_old": "cansas1d/1.0",
}


def parse_string(node: etree._Element, _version: str) -> str:
    """Access string data from a node"""
    return "".join(node.itertext())


def parse_quantity(node: etree._Element, version: str) -> Quantity[float]:
    """Pull a single quantity with length units out of an XML node"""
    magnitude = float(parse_string(node, version))
    try:
        unit = unit_parser.parse(node.attrib["unit"])
    except ValueError:
        logger.warning(
            f'Could not parse unit "{node.attrib["unit"]}".  Marking value as unitless'
        )
        unit = unitless
    return Quantity(magnitude, unit)


def attr_parse(node: etree._Element, key: str) -> str | None:
    """Parse an attribute if it is present"""
    if key in node.attrib:
        return node.attrib[key]
    return None


def opt_parse[T](
    node: etree._Element,
    key: str,
    version: str,
    subparser: Callable[[etree._Element, str], T],
) -> T | None:
    """Parse subnode if preset"""
    if (inner_node := node.find(f"{version}:{key}", ns)) is not None:
        return subparser(inner_node, version)
    return None


def all_parse[T](
    node: etree._Element,
    key: str,
    version: str,
    subparser: Callable[[etree._Element, str], T],
) -> list[T]:
    """Parse subnode if preset"""
    return [subparser(n, version) for n in node.findall(f"{version}:{key}", ns)]


def parse_vec3(node: etree._Element, version: str) -> Vec3:
    """Parse a measured 3-vector"""
    x = opt_parse(node, "x", version, parse_quantity)
    y = opt_parse(node, "y", version, parse_quantity)
    z = opt_parse(node, "z", version, parse_quantity)
    return Vec3(x=x, y=y, z=z)


def parse_rot3(node: etree._Element, version: str) -> Rot3:
    """Parse a measured rotation"""
    roll = opt_parse(node, "roll", version, parse_quantity)
    pitch = opt_parse(node, "pitch", version, parse_quantity)
    yaw = opt_parse(node, "yaw", version, parse_quantity)
    return Rot3(roll=roll, pitch=pitch, yaw=yaw)


def parse_term(node: etree._Element, version: str) -> str | Quantity[float]:
    """Parse a process term, which may be a measured quantity or a string"""
    if "unit" in node.attrib:
        return parse_quantity(node, version)
    else:
        return parse_string(node, version)

def parse_process(node: etree._Element, version: str) -> Process:
    """Parse an experimental process"""
    name = opt_parse(node, "name", version, parse_string)
    date = opt_parse(node, "date", version, parse_string)
    description = opt_parse(node, "description", version, parse_string)
    terms = {
        t.attrib["name"]: parse_term(t, version)
        for t in node.findall(f"{version}:term", ns)
    }
    notes = [parse_string(note, version) for note in node.findall(f"{version}:SASprocessnote", ns)]
    notes = [n.strip() for n in notes if n is not None and n.strip()]
    return Process(name=name, date=date, description=description, terms=terms, notes=notes)


def parse_beam_size(node: etree._Element, version: str) -> BeamSize:
    """Parse a beam size"""
    return BeamSize(name=attr_parse(node, "name"), size=parse_vec3(node, version))


def parse_source(node: etree._Element, version: str) -> Source:
    """Parse a radiation source"""
    return Source(
        radiation=opt_parse(node, "radiation", version, parse_string)
        beam_size=opt_parse(node, "beam_size", version, parse_string)
        beam_shape=opt_parse(node, "beam_shape", version, parse_beam_size)
        wavelength=opt_parse(node, "wavelength", version, parse_quantity)
        wavelength_min=opt_parse(node, "wavelength_min", version, parse_quantity)
        wavelength_max=opt_parse(node, "wavelength_max", version, parse_quantity)
        wavelength_spread=opt_parse(node, "wavelength_spread", version, parse_quantity)
    )


def parse_detector(node: etree._Element, version: str) -> Detector:
    """Parse signal detector metadata"""
    return Detector(
        name=opt_parse(node, "name", version, parse_string),
        distance=opt_parse(node, "SDD", version, parse_quantity),
        offset=opt_parse(node, "offset", version, parse_vec3),
        orientation=opt_parse(node, "orientation", version, parse_rot3),
        beam_center=opt_parse(node, "beam_center", version, parse_vec3),
        pixel_size=opt_parse(node, "pixel_size", version, parse_vec3),
        slit_length=opt_parse(node, "slit_length", version, parse_quantity),
    )


def parse_aperture(node: etree._Element, version: str) -> Aperture:
    """Parse an aperture description"""
    size = opt_parse(node, "size", version, parse_vec3)
    if size and (innerSize := node.find(f"{version}:size", ns)) is not None:
        size_name = attr_parse(innerSize, "name")
    else:
        size_name = None
    return Aperture(
        distance=opt_parse(node, "distance", version, parse_quantity),
        size=size,
        size_name=size_name,
        name=attr_parse(node, "name"),
        type_=attr_parse(node, "type"),
    )


def parse_collimation(node: etree._Element, version: str) -> Collimation:
    """Parse a beam collimation"""
    return Collimation(
        length=opt_parse(node, "length", version, parse_quantity),
        apertures=all_parse(node, "aperture", version, parse_aperture),
    )


def parse_instrument(node: etree._Element, version: str) -> Instrument:
    """Parse instrument metadata"""
    source = opt_parse(node, "SASsource", version, parse_source)
    detector = all_parse(node, "SASdetector", version, parse_detector)
    collimations = all_parse(node, "SAScollimation", version, parse_collimation)
    return Instrument(source=source, detector=detector, collimations=collimations)


def parse_sample(node: etree._Element, version: str) -> Sample:
    """Parse sample metadata"""
    return Sample(
        name=attr_parse(node, "name"),
        sample_id=opt_parse(node, "ID", version, parse_string),
        thickness=opt_parse(node, "thickness", version, parse_quantity),
        transmission=opt_parse(
            node, "transmission", version, lambda n, v: float(parse_string(n, v))
        ),
        temperature=opt_parse(node, "temperature", version, parse_quantity),
        position=opt_parse(node, "position", version, parse_vec3),
        orientation=opt_parse(node, "orientation", version, parse_rot3),
        details=all_parse(node, "details", version, parse_string),
    )


def parse_data(node: etree._Element, version: str) -> dict[str, Quantity]:
    """Parse scattering data"""
    aos = []
    keys = set()
    # Units for quantities
    us: dict[str, Unit] = {}
    for idata in node.findall(f"{version}:Idata", ns):
        struct = {}
        for value in idata.getchildren():
            name = etree.QName(value).localname
            if value.text is None or parse_string(value, version).strip() == "":
                continue
            if name not in us:
                unit = (
                    unit_parser.parse(value.attrib["unit"])
                    if "unit" in value.attrib
                    else unitless
                )
                us[name] = unit
            struct[name] = float(parse_string(value, version))
            keys.add(name)
        aos.append(struct)

    # Convert array of structures to structure of arrays
    soa: dict[str, list[float]] = {}
    for key in keys:
        soa[key] = []
    for point in aos:
        for key in keys:
            if key in point:
                soa[key].append(point[key])
            else:
                soa[key].append(np.nan)

    uncertainties = set([x for x in keys if x.endswith("dev") and x[:-3] in keys])
    keys = keys.difference(uncertainties)

    result: dict[str, Quantity] = {}
    for k in keys:
        result[k] = Quantity(np.array(soa[k]), us[k])
        if k + "dev" in uncertainties:
            result[k] = result[k].with_standard_error(
                Quantity(np.array(soa[k + "dev"]), us[k + "dev"])
            )

    return result


def get_cansas_version(root) -> str | None:
    """Find the cansas version of a file"""
    for n, v in ns.items():
        if root.tag == "{" + v + "}SASroot":
            return n
    return None


def load_data(filename: str) -> dict[str, SasData]:
    """Load scattering data from an XML file"""
    loaded_data: dict[str, SasData] = {}
    tree = etree.parse(filename)
    root = tree.getroot()

    version = get_cansas_version(root)

    if version is None:
        logger.error(f"Invalid root: {root.tag!r}")
        return loaded_data

    # How many data sets have we loaded?
    dataindex = 1

    for entry in tree.getroot().findall(f"{version}:SASentry", ns):
        name = attr_parse(entry, "name")

        title = opt_parse(entry, "Title", version, parse_string)

        if name is None:
            if title is None:
                name = f"SasData{dataindex:02}"
            else:
                name = title

        metadata = Metadata(
            title=title,
            run=all_parse(entry, "Run", version, parse_string),
            instrument=opt_parse(entry, "SASinstrument", version, parse_instrument),
            process=all_parse(entry, "SASprocess", version, parse_process),
            sample=opt_parse(entry, "SASsample", version, parse_sample),
            definition=opt_parse(entry, "SASdefinition", version, parse_string),
        )

        data = {}

        datacount = 0
        for n in entry.findall(f"{version}:SASdata", ns):
            datacount += 1
            data_set = parse_data(n, version)
            data = data_set
            break

        loaded_data[name] = SasData(
            name=name,
            dataset_type=one_dim,
            data_contents=data,
            metadata=metadata,
            verbose=False,
        )
        dataindex += 1
    return loaded_data


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = [test_file]

    for f in files:
        print(f)
        data = load_data(f)

        for dataset in data.values():
            print(dataset.summary())
