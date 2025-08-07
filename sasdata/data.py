
import numpy as np

from sasdata import dataset_types
from sasdata.dataset_types import DatasetType
from sasdata.metadata import Metadata
from sasdata.quantities.quantity import Quantity


class SasData:
    def __init__(self, name: str,
                 data_contents: dict[str, Quantity],
                 dataset_type: DatasetType,
                 metadata: Metadata,
                 verbose: bool=False):

        self.name = name
        # validate data contents
        if not all([key in dataset_type.optional or key in dataset_type.required for key in data_contents]):
            raise ValueError("Columns don't match the dataset type")
        self._data_contents = data_contents
        self._verbose = verbose

        self.metadata = metadata

        # TODO: Could this be optional?
        self.dataset_type: DatasetType = dataset_type

        # Components that need to be organised after creation
        self.mask = None # TODO: fill out
        self.model_requirements = None # TODO: fill out

    # TODO: Handle the other data types.
    @property
    def ordinate(self) -> Quantity:
        match self.dataset_type:
            case dataset_types.one_dim | dataset_types.two_dim:
                return self._data_contents["I"]
            case dataset_types.sesans:
                return self._data_contents["Depolarisation"]
            case _:
                return None

    @property
    def abscissae(self) -> Quantity:
        match self.dataset_type:
            case dataset_types.one_dim:
                return self._data_contents['Q']
            case dataset_types.two_dim:
                # Type hinting is a bit lacking. Assume each part of the zip is a scalar value.
                data_contents = np.array(list(zip(self._data_contents['Qx'].value, self._data_contents['Qy'].value)))
                # Use this value to extract units etc. Assume they will be the same for Qy.
                reference_data_content = self._data_contents['Qx']
                # TODO: If this is a derived quantity then we are going to lose that
                # information.
                #
                # TODO: Won't work when there's errors involved. On reflection, we
                # probably want to avoid creating a new Quantity but at the moment I
                # can't see a way around it.
                return Quantity(data_contents, reference_data_content.units)
            case dataset_types.sesans:
                return self._data_contents["SpinEchoLength"]
            case _:
                None

    def __getitem__(self, item: str):
        return self._data_contents[item]

    def summary(self, indent = "  "):
        s = f"{self.name}\n"

        for data in sorted(self._data_contents, reverse=True):
            s += f"{indent}{data}\n"

        s += "Metadata:\n"
        s += "\n"
        s += self.metadata.summary()

        return s
