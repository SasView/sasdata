import copy
from math import fabs, sqrt
from typing import Union

import numpy as np

from sasdata.data.plottables import Plottable, PlottableMeta
from sasdata.data.data_info import DataInfo
from sasdata.data_util.uncertainty import Uncertainty
from sasdata.data_util.deprecation import deprecated

# TODO: Remove top-level Data1D and Data2D class - only keep overarching Data class
#  - Use isinstance for data typing(?)
#  - Move clone, copy, etc., into main Data class
#  - Typing hints


# FIXME: Metaclass here, not what I currently have
class Data(DataInfo, metaclass=PlottableMeta):

    _plottable = None
    _data_info = None

    """Single top-level class for all Data objects"""
    def __init__(self, plottable: Plottable, data_info: DataInfo):
        Plottable.__init__(self, plottable.x, plottable.y)
        DataInfo.__init__(self)
        self._plottable = plottable
        self._data_info = data_info
        # Create class-level properties for each sub-class property
        for key, value in plottable.__dict__:
            Data.key = property(value)
        for key, value in data_info.__dict__:
            Data.key = property(value)

    def __str__(self) -> str:
        """
        Nice printout
        """
        return f"{self._data_info.__str__()}\n{self._plottable.__str__()}"

    def is_slit_smeared(self) -> bool:
        """
        Check whether the data has slit smearing information
        :return: True is slit smearing info is present, False otherwise
        """
        def _check(obj: Data, param: str):
            val = getattr(obj, param, None)
            return (hasattr(obj, param) and val is not None and any(val))
        return _check(self, 'dxl') or _check(self, 'dxw')

    def clone_without_data(self, length: int = 0, clone: Data = None) -> Data:
        """
        Clone the current object, without copying the data (which
        will be filled out by a subsequent operation).
        The data arrays will be initialized to zero.

        :param length: length of the data array to be initialized
        :param clone: if provided, the data will be copied to clone
        """
        from copy import deepcopy

        if clone is None or not issubclass(clone.__class__, Data):
            x = np.zeros(length)
            y = np.zeros(length)
            # Determine the class of the plottable and create a dummy instance of that class type
            plottable = self._plottable.__class__(x, y)
            data_info = DataInfo()
            clone = Data(plottable, data_info)

        for key, value in self._plottable.__dict__:
            clone.key = property(deepcopy(value))

        return clone

    @deprecated(replaced_with="Data.assign_data_from_plottable")
    def copy_from_datainfo(self, data1d: Union[Data, Plottable]):
        self.assign_data_from_plottable(data1d)

    def assign_data_from_plottable(self, plottable: Plottable):
        """
        copy values of Plottable type, ensuring all sub-classes are captured
        """
        for key, value in plottable.__dict__:
            setattr(self, key, value)

    def _validity_check(self, other: Union[Data, Plottable]) -> (np.array, np.array):
        """
        Checks that the data lengths are compatible.
        Checks that the x vectors are compatible.
        Returns errors vectors equal to original
        errors vectors if they were present or vectors
        of zeros when none was found.

        :param other: other data set for operation
        :return: dy for self, dy for other [numpy arrays]
        :raise ValueError: when lengths are not compatible
        """
        dy_other = None
        for key, val in other.__dict__:
            if not hasattr(self, key) or not isinstance(getattr(self, key).__class__(), val):
                raise ValueError(f'Unable to perform operation: values in {key} are not compatible.')

        # Check that we have errors, otherwise create zero vector
        dy = self.dy
        if self.dy is None or (len(self.dy) != len(self.y)):
            dy = np.zeros(len(self.y))

        return dy, dy_other

    def _perform_operation(self, other: Union[Data, Plottable], operation: str):
        """
        # TODO: documentation!!!!
        """
        # First, check the data compatibility
        dy, dy_other = self._validity_check(other)
        result = self.clone_without_data(len(self.x))
        if self.dxw is None:
            result.dxw = None
        else:
            result.dxw = np.zeros(len(self.x))
        if self.dxl is None:
            result.dxl = None
        else:
            result.dxl = np.zeros(len(self.x))

        for i in range(len(self.x)):
            result.x[i] = self.x[i]
            if self.dx is not None and len(self.x) == len(self.dx):
                result.dx[i] = self.dx[i]
            if self.dxw is not None and len(self.x) == len(self.dxw):
                result.dxw[i] = self.dxw[i]
            if self.dxl is not None and len(self.x) == len(self.dxl):
                result.dxl[i] = self.dxl[i]

            a = Uncertainty(self.y[i], dy[i]**2)
            if isinstance(other, Data1D):
                b = Uncertainty(other.y[i], dy_other[i]**2)
                if other.dx is not None:
                    result.dx[i] *= self.dx[i]
                    result.dx[i] += (other.dx[i]**2)
                    result.dx[i] /= 2
                    result.dx[i] = sqrt(result.dx[i])
                if result.dxl is not None and other.dxl is not None:
                    result.dxl[i] *= self.dxl[i]
                    result.dxl[i] += (other.dxl[i]**2)
                    result.dxl[i] /= 2
                    result.dxl[i] = sqrt(result.dxl[i])
            else:
                b = other

            output = operation(a, b)
            result.y[i] = output.x
            result.dy[i] = sqrt(fabs(output.variance))
        return result

    def _validity_check_union(self, other):
        """
        Checks that the data lengths are compatible.
        Checks that the x vectors are compatible.
        Returns errors vectors equal to original
        errors vectors if they were present or vectors
        of zeros when none was found.

        :param other: other data set for operation
        :return: bool
        :raise ValueError: when data types are not compatible
        """
        # FIXME: Get type of linked plottable
        if not isinstance(other, self._plottable.__class__()):
            msg = "Unable to perform operation: different types of data set"
            raise ValueError(msg)
        return True

    def _perform_union(self, other):
        """
        """
        # First, check the data compatibility
        # TODO: Abstract, abstract, abstract...
        self._validity_check_union(other)
        result = self.clone_without_data(len(self.x) + len(other.x))
        if self.dy is None or other.dy is None:
            result.dy = None
        else:
            result.dy = np.zeros(len(self.x) + len(other.x))
        if self.dx is None or other.dx is None:
            result.dx = None
        else:
            result.dx = np.zeros(len(self.x) + len(other.x))
        if self.dxw is None or other.dxw is None:
            result.dxw = None
        else:
            result.dxw = np.zeros(len(self.x) + len(other.x))
        if self.dxl is None or other.dxl is None:
            result.dxl = None
        else:
            result.dxl = np.zeros(len(self.x) + len(other.x))

        result.x = np.append(self.x, other.x)
        # argsorting
        ind = np.argsort(result.x)
        result.x = result.x[ind]
        result.y = np.append(self.y, other.y)
        result.y = result.y[ind]
        if result.dy is not None:
            result.dy = np.append(self.dy, other.dy)
            result.dy = result.dy[ind]
        if result.dx is not None:
            result.dx = np.append(self.dx, other.dx)
            result.dx = result.dx[ind]
        if result.dxw is not None:
            result.dxw = np.append(self.dxw, other.dxw)
            result.dxw = result.dxw[ind]
        if result.dxl is not None:
            result.dxl = np.append(self.dxl, other.dxl)
            result.dxl = result.dxl[ind]
        return result


@deprecated(replaced_with="sasdata.data.data.Data(Plottable, DataInfo)")
def combine_data_info_with_plottable(data: Plottable, datainfo: DataInfo):
    """
    A function that combines the DataInfo data in self.current_datainto with a
    plottable_1D or 2D data object.

    :param data: A plottable_1D or plottable_2D data object
    :param datainfo: A DataInfo object to be combined with the plottable
    :return: A fully specified Data1D or Data2D object
    """

    return Data(data, datainfo)
