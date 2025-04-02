import logging
from lxml import etree

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

        run = title = None
        title = _load_text(entry, "Title")
        run = _load_text(entry, "Run")

        processes = [
            parse_process(node) for node in entry.findall("cansas:SASprocess", ns)
        ]

        metadata = Metadata(
            title=title,
            run=[run] if run is not None else [],
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
