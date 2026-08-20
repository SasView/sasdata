import json
import typing
from typing import Any

import h5py
import numpy as np
from h5py._hl.group import Group as HDF5Group

from sasdata import dataset_types
from sasdata.abscissa import Abscissa
from sasdata.dataset_types import DatasetType
from sasdata.metadata import DerivedMetadata, Metadata, MetadataEncoder
from sasdata.quantities.quantity import Quantity


class SasData:
    """General object containing data in the SasView ecosystem"""

    def __init__(
        self,
        name: str,
        ordinate: Quantity,
        abscissae: Abscissa,
        mask: Quantity,
        dependents: list["SasData"],
        metadata: Metadata,
    ):
        self.name = name
        self._ordinate = ordinate
        self._abscissae = abscissae
        self._mask = mask
        self.dependents = dependents
        self.metadata = metadata

    @property
    def ordinate(self) -> Quantity:
        return self._ordinate

    @property
    def abscissae(self) -> Abscissa:
        return self._abscissae

    @property
    def mask(self) -> Quantity:
        return self._mask

    def scatter_data(self):
        """Return data in the coordinate/value form [(x1, x2, x3, y)...]"""


class SasDerivedMeasurement(SasData):
    """General object sas measurement that has not come directly from a file,
    for example, the difference between two datasets"""

    def __init__(
        self,
        name: str,
        ordinate: Quantity,
        abscissae: Abscissa,
        mask: Quantity,
        dependents: list["SasData"],
        metadata: DerivedMetadata,
    ):
        super().__init__(
            name=name, ordinate=ordinate, abscissae=abscissae, mask=mask, dependents=dependents, metadata=metadata
        )


class SasMeasurement(SasData):
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
        self._mask = None  # TODO: fill out
        self.model_requirements = None  # TODO: fill out

    # TODO: Handle the other data types.
    @property
    def ordinate(self) -> Quantity:
        match self.dataset_type:
            case dataset_types.one_dim | dataset_types.two_dim | dataset_types.three_dim | dataset_types.angle_dim:
                return self._data_contents["I"]
            case dataset_types.sesans:
                return self._data_contents["Depolarisation"]
            case _:
                return None

    @property
    def abscissae(self) -> Abscissa:
        match self.dataset_type:
            case dataset_types.one_dim:
                return Abscissa.determine([self._data_contents["Q"]], self.ordinate)
            case dataset_types.two_dim:
                return Abscissa.determine([self._data_contents["Qx"], self._data_contents["Qy"]], self.ordinate)
            case dataset_types.angle_dim:
                return Abscissa.determine([self._data_contents["Phi"]], self.ordinate)
            case dataset_types.three_dim:
                return Abscissa.determine(
                    [self._data_contents["Qx"], self._data_contents["Qy"], self._data_contents["Qz"]], self.ordinate
                )
            case dataset_types.sesans:
                return Abscissa.determine([self._data_contents["SpinEchoLength"]], self.ordinate)
            case _:
                return None

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
        return SasMeasurement(
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
        sasentry.attrs["name"] = self.name
        self.metadata.as_h5(sasentry)

        # We export each data set into its own entry, so we only ever
        # need sasdata01
        group = sasentry.create_group("sasdata01")
        for idx, (key, sasdata) in enumerate(self._data_contents.items()):
            sasdata.as_h5(group, key)

    @staticmethod
    def save_h5(data: dict[str, typing.Self], path: str | typing.BinaryIO):
        with h5py.File(path, "w") as f:
            for idx, (key, data) in enumerate(data.items()):
                sasentry = f.create_group(f"sasentry{idx + 1:02d}")
                if not key.startswith("sasentry"):
                    sasentry.attrs["sasview_key"] = key
                data._save_h5(sasentry)

    @staticmethod
    def deserialise(data: str) -> "SasData":
        json_data = json.loads(data)
        return SasData.deserialise_json(json_data)

    @staticmethod
    def deserialise_json(json_data: dict) -> "SasData":
        name = json_data["name"]
        data_contents = {}
        dataset_type = json_data["dataset_type"]  # TODO: update when DatasetType is more finalized
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
            "dataset_type": None,  # TODO: update when DatasetType is more finalized
            "verbose": self._verbose,
            "metadata": self.metadata.serialise_json(),
            "mask": {},
            "model_requirements": {},
        }


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
                    "data_contents": obj._data_contents,
                    "type": obj.dataset_type,
                    "mask": obj.mask,
                    "metadata": obj.metadata,
                    "model_requirements": obj.model_requirements,
                }
            case _:
                return super().default(obj)


def sasdata_reader2D_converter(data2d: SasData | None = None) -> SasData:
    """
    convert old 2d format opened by IhorReader or danse_reader
    to new 2D SasData format
    This is mainly used by the Readers

    :param data2d: SasData object with 2D arrays
    :return: SasData object with 1D arrays

    """
    if data2d._data_contents["I"] is None or data2d.x_bins is None or data2d.y_bins is None:
        raise ValueError("Can't convert this data: data=None...")
    new_x = np.tile(data2d.x_bins, (len(data2d.y_bins), 1))
    new_y = np.tile(data2d.y_bins, (len(data2d.x_bins), 1))
    new_y = new_y.swapaxes(0, 1)

    new_data = data2d._data_contents["I"].value.flatten()
    qx_data = new_x.flatten()
    qy_data = new_y.flatten()
    err_data = np.sqrt(data2d._data_contents["I"].variance.value)
    if not data2d._data_contents["I"].has_error or np.any(err_data <= 0):
        new_err_data = np.sqrt(np.abs(new_data))
    else:
        new_err_data = err_data.flatten()
    mask = np.ones(len(new_data), dtype=bool)

    data2d._data_contents["I"].value = new_data
    data2d._data_contents["I"].variance.value = new_err_data**2
    data2d._data_contents["Qx"].value = qx_data
    data2d._data_contents["Qy"].value = qy_data
    data2d._mask = mask

    return data2d
