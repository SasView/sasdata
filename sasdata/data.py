from dataclasses import dataclass
from quantities.quantities import Quantity, NamedQuantity
from sasdata.metadata import MetaData

import numpy as np

from sasdata.model_requirements import ModellingRequirements


class SasData:
    def __init__(self, name: str,
                 data_contents: list[Quantity],
                 raw_metadata: Group,
                 instrument: Instrument,
                 verbose: bool=False):

        self.name = name
        self._data_contents = data_contents
        self._raw_metadata = raw_metadata
        self._verbose = verbose

        self.metadata = Metadata(AccessorTarget(raw_metadata, verbose=verbose), instrument)

        # Components that need to be organised after creation
        self.mask = None # TODO: fill out
        self.model_requirements = None # TODO: fill out

    #TODO: This seems oriented around 1D I vs Q data. What about 2D data?
    @property
    def ordinate() -> Quantity:
        raise NotImplementedError()

    @property
    def abscissae(self) -> Quantity:
        if self.dataset_type == one_dim:
            return self._data_contents['Q']
        elif self.dataset_type == two_dim:
            # Type hinting is a bit lacking. Assume each part of the zip is a scalar value.
            data_contents = zip(self._data_contents['Qx'].value, self._data_contents['Qy'].value)
            # Use this value to extract units etc. Assume they will be the same for Qy.
            reference_data_content = self._data_contents['Qx']
            # TODO: If this is a derived quantity then we are going to lose that
            # information.
            #
            # TODO: Won't work when there's errors involved. On reflection, we
            # probably want to avoid creating a new Quantity but at the moment I
            # can't see a way around it.
            return Quantity(data_contents, reference_data_content.units)
        return None

    def __getitem__(self, item: str):
        return self._data_contents[item]

    def summary(self, indent = "  ", include_raw=False):
        s = f"{self.name}\n"

        for data in self._data_contents:
            s += f"{indent}{data}\n"

        s += f"Metadata:\n"
        s += "\n"
        s += self.metadata.summary()

        if include_raw:
            s += key_tree(self._raw_metadata)

        return s

    @staticmethod
    def deserialise(data: str) -> "SasData":
        json_data = json.loads(data)
        return SasData.deserialise_json(json_data)

    @staticmethod
    def deserialise_json(json_data: dict) -> "SasData":
        name = json_data["name"]
        data_contents = {}
        dataset_type = json_data["dataset_type"] # TODO: update when DatasetType is more finalized
        metadata = json_data["metadata"].deserialise_json()
        for quantity in json_data["data_contents"]:
            data_contents[quantity["label"]] = Quantity.deserialise_json(quantity)
        return SasData(name, data_contents, dataset_type, metadata)

    def serialise(self) -> str:
        return json.dumps(self._serialise_json())

    # TODO: fix serializers eventually
    def _serialise_json(self) -> dict[str, Any]:
        data = []
        for d in self._data_contents:
            quantity = self._data_contents[d]
            quantity["label"] = d
            data.append(quantity)
        return {
            "name": self.name,
            "data_contents": data,
            "dataset_type": None, # TODO: update when DatasetType is more finalized
            "verbose": self._verbose,
            "metadata": self.metadata.serialise_json(),
            "mask": {},
            "model_requirements": {}
        }
