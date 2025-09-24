import typing

import h5py
from h5py._hl.group import Group as HDF5Group
import numpy as np

from sasdata import dataset_types
from sasdata.dataset_types import DatasetType
from sasdata.metadata import Metadata, MetadataEncoder
from sasdata.quantities.quantity import Quantity


class SasData:
    def __init__(
        self,
        name: str,
        data_contents: dict[str, Quantity],
        dataset_type: DatasetType,
        metadata: Metadata,
        verbose: bool = False,
    ):
        self.name = name
        # validate data contents
        if not all([key in dataset_type.optional or key in dataset_type.required for key in data_contents]):
            raise ValueError(f"Columns don't match the dataset type: {[key for key in data_contents]}")
        self._data_contents = data_contents
        self._verbose = verbose

        self.metadata = metadata

        # TODO: Could this be optional?
        self.dataset_type: DatasetType = dataset_type

        # Components that need to be organised after creation
        self.mask = None  # TODO: fill out
        self.model_requirements = None  # TODO: fill out

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
                return self._data_contents["Q"]
            case dataset_types.two_dim:
                # Type hinting is a bit lacking. Assume each part of the zip is a scalar value.
                data_contents = np.array(
                    list(
                        zip(
                            self._data_contents["Qx"].value,
                            self._data_contents["Qy"].value,
                        )
                    )
                )
                # Use this value to extract units etc. Assume they will be the same for Qy.
                reference_data_content = self._data_contents["Qx"]
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

    def summary(self, indent="  "):
        s = f"{self.name}\n"

        for data in sorted(self._data_contents, reverse=True):
            s += f"{indent}{data}\n"

        s += "Metadata:\n"
        s += "\n"
        s += self.metadata.summary()

        return s

    @staticmethod
    def from_json(obj):
        return SasData(
            name=obj["name"],
            dataset_type=DatasetType(
                name=obj["type"]["name"],
                required=obj["type"]["required"],
                optional=obj["type"]["optional"],
                expected_orders=obj["type"]["expected_orders"],
            ),
            data_contents=obj["data_contents"],
            metadata=Metadata.from_json(obj["metadata"]),
        )

    def _save_h5(self, sasentry: HDF5Group):
        """Export data into HDF5 file"""
        # with h5py.File(path, "w") as f:
        sasentry.attrs["name"] = self.name
        self.metadata.as_h5(sasentry)
        group = sasentry.create_group("sasdata01")
        for idx, (key, sasdata) in enumerate(self._data_contents.items()):
            sasdata.as_h5(group, key)


    @staticmethod
    def save_h5(data: dict[str, typing.Self], path: str | typing.BinaryIO):
        with h5py.File(path, "w") as f:
            for idx, (key, data) in enumerate(data.items()):
                sasentry = f.create_group(f"sasentry{idx+1:02d}")
                if not key.startswith("sasentry"):
                    sasentry.attrs["sasview_key"] = key
                data._save_h5(sasentry)



class SasDataEncoder(MetadataEncoder):
    def default(self, obj):
        match obj:
            case DatasetType():
                return {
                    "name": obj.name,
                    "required": obj.required,
                    "optional": obj.optional,
                    "expected_orders": obj.expected_orders,
                }
            case SasData():
                return {
                    "name": obj.name,
                    "data_contents": {},
                    "type": obj.dataset_type,
                    "mask": obj.mask,
                    "metadata": obj.metadata,
                    "model_requirements": obj.model_requirements,
                }
            case _:
                return super().default(obj)
