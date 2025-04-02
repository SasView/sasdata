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
from sasdata.quantities.units import Unit

logger = logging.getLogger(__name__)

test_file = "./example_data/1d_data/ISIS_Polymer_Blend_TK49.xml"

ns = {"cansas": "urn:cansas1d:1.1"}


# Small helper function to optionally load child text from an element
def _load_text(node: etree.Element, name: str) -> str | None:
    if (inner_node := node.find(f"cansas:{name}", ns)) is not None:
        return inner_node.text
    return None


def parse_string(node: etree.Element) -> str:
    """Access string data from a node"""
    return node.text


def parse_quantity(node: etree.Element) -> Quantity[float]:
    """Pull a single quantity with length units out of an XML node"""
    magnitude = float(node.text)
    unit = node.attrib["unit"]
    return Quantity(magnitude, unit_parser.parse(unit))


def attr_parse(node: etree.Element, key: str) -> str | None:
    """Parse an attribute if it is present"""
    if key in node.attrib:
        return node.attrib[key]
    return None


def opt_parse[T](
    node: etree.Element, key: str, subparser: Callable[[etree.Element], T]
) -> T | None:
    """Parse subnode if preset"""
    if (inner_node := node.find(f"cansas:{key}", ns)) is not None:
        return subparser(inner_node)
    return None


def all_parse[T](
    node: etree.Element, key: str, subparser: Callable[[etree.Element], T]
) -> list[T]:
    """Parse subnode if preset"""
    return [subparser(n) for n in node.findall(f"cansas:{key}", ns)]
    if (inner_node := node.find(f"cansas:{key}", ns)) is not None:
        return subparser(inner_node)
    return None


def parse_vec3(node: etree.Element) -> Vec3:
    """Parse a measured 3-vector"""
    x = opt_parse(node, "x", parse_quantity)
    y = opt_parse(node, "y", parse_quantity)
    z = opt_parse(node, "z", parse_quantity)
    return Vec3(x=x, y=y, z=z)


def parse_rot3(node: etree.Element) -> Rot3:
    """Parse a measured rotation"""
    roll = opt_parse(node, "roll", parse_quantity)
    pitch = opt_parse(node, "pitch", parse_quantity)
    yaw = opt_parse(node, "yaw", parse_quantity)
    return Rot3(roll=roll, pitch=pitch, yaw=yaw)


def parse_process(node: etree.Element) -> Process:
    name = _load_text(node, "name")
    date = _load_text(node, "date")
    description = _load_text(node, "description")
    terms = {t.attrib["name"]: t.text for t in node.findall("cansas:term", ns)}
    return Process(name=name, date=date, description=description, term=terms)


def parse_beam_size(node: etree.Element) -> BeamSize:
    return BeamSize(
        name=opt_parse(node, "name", parse_string),
        size=opt_parse(node, "size", parse_vec3),
    )


def parse_source(node: etree.Element) -> Source:
    radiation = opt_parse(node, "radiation", parse_string)
    beam_shape = opt_parse(node, "beam_shape", parse_string)
    beam_size = opt_parse(node, "beam_size", parse_beam_size)
    wavelength = opt_parse(node, "wavelength", parse_quantity)
    wavelength_min = opt_parse(node, "wavelength_min", parse_quantity)
    wavelength_max = opt_parse(node, "wavelength_max", parse_quantity)
    wavelength_spread = opt_parse(node, "wavelength_spread", parse_quantity)
    return Source(
        radiation=radiation,
        beam_size=beam_size,
        beam_shape=beam_shape,
        wavelength=wavelength,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_spread=wavelength_spread,
    )


def parse_detector(node: etree.Element) -> Detector:
    return Detector(
        name=opt_parse(node, "name", parse_string),
        distance=opt_parse(node, "SDD", parse_quantity),
        offset=opt_parse(node, "offset", parse_vec3),
        orientation=opt_parse(node, "orientation", parse_rot3),
        beam_center=opt_parse(node, "beam_center", parse_vec3),
        pixel_size=opt_parse(node, "pixel_size", parse_vec3),
        slit_length=opt_parse(node, "slit_length", parse_quantity),
    )


def parse_aperture(node: etree.Element) -> Aperture:
    size = opt_parse(node, "size", parse_vec3)
    if size:
        size_name = attr_parse(node["size"], "name")
    else:
        size_name = None
    return Aperture(
        distance=opt_parse(node, "distance", parse_quantity),
        size=size,
        size_name=size_name,
        name=attr_parse(node, "name"),
        type_=attr_parse(node, "type"),
    )


def parse_collimation(node: etree.Element) -> Collimation:
    return Collimation(
        length=opt_parse(node, "length", parse_quantity),
        apertures=all_parse(node, "aperture", parse_aperture),
    )


def parse_instrument(node: etree.Element) -> Instrument:
    source = opt_parse(node, "SASsource", parse_source)
    detector = all_parse(node, "SASdetector", parse_detector)
    collimations = all_parse(node, "SAScollimation", parse_collimation)
    return Instrument(source=source, detector=detector, collimations=collimations)


def parse_sample(node: etree.Element) -> Sample:
    return Sample(
        name=attr_parse(node, "name"),
        sample_id=opt_parse(node, "ID", parse_string),
        thickness=opt_parse(node, "thickness", parse_quantity),
        transmission=opt_parse(node, "transmission", lambda n: float(parse_string(n))),
        temperature=opt_parse(node, "temperature", parse_quantity),
        position=opt_parse(node, "position", parse_vec3),
        orientation=opt_parse(node, "orientation", parse_rot3),
        details=all_parse(node, "details", parse_string),
    )


def parse_data(node: etree.Element) -> dict[str, Quantity]:
    aos = []
    keys = set()
    # Units for quantities
    us: dict[str, Unit] = {}
    for idata in node.findall("cansas:Idata", ns):
        struct = {}
        for value in idata.getchildren():
            name = etree.QName(value).localname
            if name not in us:
                unit = unit_parser.parse(value.attrib["unit"])
                us[name] = unit
            struct[name] = float(value.text)
            keys.add(name)
        aos.append(struct)

    # Convert array of structures to strucgture of arrays
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


def load_data(filename) -> dict[str, SasData]:
    loaded_data: dict[str, SasData] = {}
    tree = etree.parse(filename)
    root = tree.getroot()
    if root.tag != "{" + ns["cansas"] + "}SASroot":
        logger.error(f"Invalid root: {root.tag}")
        return loaded_data
    for entry in tree.getroot().findall("cansas:SASentry", ns):
        name = entry.attrib["name"]

        metadata = Metadata(
            title=opt_parse(entry, "Title", parse_string),
            run=all_parse(entry, "Run", parse_string),
            instrument=opt_parse(entry, "SASinstrument", parse_instrument),
            process=all_parse(entry, "SASprocess", parse_process),
            sample=opt_parse(entry, "SASsample", parse_sample),
            definition=opt_parse(entry, "SASdefinition", parse_string),
        )

        data = {}

        datacount = 0
        for n in entry.findall("cansas:SASdata", ns):
            datacount += 1
            data_set = parse_data(n)
            data = data_set
            break

        loaded_data[name] = SasData(
            name=name,
            dataset_type=one_dim,
            data_contents=data,
            metadata=metadata,
            verbose=False,
        )
    return loaded_data


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = [test_file]

    for f in files:
        try:
            data = load_data(f)

            for dataset in data.values():
                print(dataset.summary())
        except:
            print(f)
