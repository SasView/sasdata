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
            title="",
            run=[],
            instrument=None,
            process=[],
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
