import logging
from lxml import etree
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


def parse_process(node: etree.Element) -> Process:
    name = _load_text(node, "name")
    date = _load_text(node, "date")
    description = _load_text(node, "description")
    terms = {t.attrib["name"]: t.text for t in node.findall("cansas:term", ns)}
    return Process(name=name, date=date, description=description, term=terms)


def load_data(filename) -> dict[str, SasData]:
    loaded_data: dict[str, SasData] = {}
    tree = etree.parse(filename)
    root = tree.getroot()
    if root.tag != "{" + ns["cansas"] + "}SASroot":
        logger.error(f"Invalid root: {root.tag}")
        return loaded_data
    for entry in tree.getroot().findall("cansas:SASentry", ns):
        name = entry.attrib["name"]

        title = _load_text(entry, "Title")
        runs = all_parse(entry, "Run", parse_string)

        processes = all_parse(entry, "SASprocess", parse_process)

        metadata = Metadata(
            title=title,
            run=runs,
            instrument=None,
            process=processes,
            sample=None,
            definition=None,
        )
        loaded_data[name] = SasData(
            name=name,
            dataset_type=one_dim,
            data_contents={},
            metadata=metadata,
            verbose=False,
        )
    return loaded_data


if __name__ == "__main__":
    data = load_data(test_file)

    for dataset in data.values():
        print(dataset.summary())
