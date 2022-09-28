import copy
from math import fabs, sqrt

import numpy as np

from sasdata.data.plottables import Plottable, Plottable1D, Plottable2D
from sasdata.data.data_info import DataInfo
from sasdata.data_util.uncertainty import Uncertainty


class Data(Plottable, DataInfo):
    """Abstract base class all Data objects should inherit from """
    def __init__(self, x=None, y=None, dx=None, dy=None, mask=None):
        Plottable.__init__(self, x, y, dx, dy, mask)
        DataInfo.__init__(self)


class Data1D(Data):
    """1D data class"""

    def __init__(self, x=None, y=None, dx=None, dy=None, mask=None):
        super().__init__(x, y, dx, dy, mask)

    def __str__(self):
        """
        Nice printout
        """
        _str = "%s\n" % DataInfo.__str__(self)
        _str += "Data:\n"
        _str += "   Type:         %s\n" % self.__class__.__name__
        _str += "   X-axis:       %s\t[%s]\n" % (self._xaxis, self._xunit)
        _str += "   Y-axis:       %s\t[%s]\n" % (self._yaxis, self._yunit)
        _str += "   Length:       %g\n" % len(self.x)
        return _str

    def is_slit_smeared(self):
        """
        Check whether the data has slit smearing information
        :return: True is slit smearing info is present, False otherwise
        """
        def _check(v):
            return ((v.__class__ == list or v.__class__ == np.ndarray)
                    and len(v) > 0 and min(v) > 0)
        return _check(self.dxl) or _check(self.dxw)

    def clone_without_data(self, length=0, clone=None):
        """
        Clone the current object, without copying the data (which
        will be filled out by a subsequent operation).
        The data arrays will be initialized to zero.

        :param length: length of the data array to be initialized
        :param clone: if provided, the data will be copied to clone
        """
        from copy import deepcopy

        if clone is None or not issubclass(clone.__class__, Data1D):
            x = np.zeros(length)
            dx = np.zeros(length)
            y = np.zeros(length)
            dy = np.zeros(length)
            lam = np.zeros(length)
            dlam = np.zeros(length)
            clone = Data1D(x, y, lam=lam, dx=dx, dy=dy, dlam=dlam)

        clone.title = self.title
        clone.run = self.run
        clone.filename = self.filename
        clone.instrument = self.instrument
        clone.notes = deepcopy(self.notes)
        clone.process = deepcopy(self.process)
        clone.detector = deepcopy(self.detector)
        clone.sample = deepcopy(self.sample)
        clone.source = deepcopy(self.source)
        clone.collimation = deepcopy(self.collimation)
        clone.trans_spectrum = deepcopy(self.trans_spectrum)
        clone.meta_data = deepcopy(self.meta_data)
        clone.errors = deepcopy(self.errors)

        return clone

    def copy_from_datainfo(self, data1d):
        """
        copy values of Data1D of type DataLaoder.Data_info
        """
        self.x  = copy.deepcopy(data1d.x)
        self.y  = copy.deepcopy(data1d.y)
        self.dy = copy.deepcopy(data1d.dy)

        if hasattr(data1d, "dx"):
            self.dx = copy.deepcopy(data1d.dx)
        if hasattr(data1d, "dxl"):
            self.dxl = copy.deepcopy(data1d.dxl)
        if hasattr(data1d, "dxw"):
            self.dxw = copy.deepcopy(data1d.dxw)

        self.xaxis(data1d._xaxis, data1d._xunit)
        self.yaxis(data1d._yaxis, data1d._yunit)
        self.title = data1d.title

    def _validity_check(self, other):
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
        if isinstance(other, Data1D):
            # Check that data lengths are the same
            if len(self.x) != len(other.x) or len(self.y) != len(other.y):
                msg = "Unable to perform operation: data length are not equal"
                raise ValueError(msg)
            # Here we could also extrapolate between data points
            TOLERANCE = 0.01
            for i in range(len(self.x)):
                if fabs(self.x[i] - other.x[i]) > self.x[i]*TOLERANCE:
                    msg = "Incompatible data sets: x-values do not match"
                    raise ValueError(msg)

            # Check that the other data set has errors, otherwise
            # create zero vector
            dy_other = other.dy
            if other.dy is None or (len(other.dy) != len(other.y)):
                dy_other = np.zeros(len(other.y))

        # Check that we have errors, otherwise create zero vector
        dy = self.dy
        if self.dy is None or (len(self.dy) != len(self.y)):
            dy = np.zeros(len(self.y))

        return dy, dy_other

    def _perform_operation(self, other, operation):
        """
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
        if not isinstance(other, Data1D):
            msg = "Unable to perform operation: different types of data set"
            raise ValueError(msg)
        return True

    def _perform_union(self, other):
        """
        """
        # First, check the data compatibility
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


class Data2D(Plottable2D, Data):
    """2D data class"""

    # Units for Q-values
    Q_unit = '1/A'
    # Units for I(Q) values
    I_unit = '1/cm'
    # No 2D SESANS data as of yet. Always set it to False
    isSesans = False

    def __init__(self, data=None, err_data=None, qx_data=None,
                 qy_data=None, q_data=None, mask=None,
                 dqx_data=None, dqy_data=None):
        DataInfo.__init__(self)
        Plottable2D.__init__(self, z=data, dz=err_data, x=qx_data, y=qy_data,
                              dx=dqx_data, dy=dqy_data, mask=mask)

        if len(self.detector) > 0:
            raise RuntimeError("Data2D: Detector bank already filled at init")

    def __str__(self):
        _str = "%s\n" % DataInfo.__str__(self)
        _str += "Data:\n"
        _str += "   Type:         %s\n" % self.__class__.__name__
        _str += "   X-axis:       %s\t[%s]\n" % (self._xaxis, self._xunit)
        _str += "   Y-axis:       %s\t[%s]\n" % (self._yaxis, self._yunit)
        _str += "   Z-axis:       %s\t[%s]\n" % (self._zaxis, self._zunit)
        _str += "   Length:       %g \n" % (len(self.data))
        _str += "   Shape:        (%d, %d)\n" % (len(self.y_bins),
                                                 len(self.x_bins))
        return _str

    def clone_without_data(self, length=0, clone=None):
        """
        Clone the current object, without copying the data (which
        will be filled out by a subsequent operation).
        The data arrays will be initialized to zero.

        :param length: length of the data array to be initialized
        :param clone: if provided, the data will be copied to clone
        """
        from copy import deepcopy

        if clone is None or not issubclass(clone.__class__, Data2D):
            data = np.zeros(length)
            err_data = np.zeros(length)
            qx_data = np.zeros(length)
            qy_data = np.zeros(length)
            q_data = np.zeros(length)
            mask = np.zeros(length)
            clone = Data2D(data=data, err_data=err_data,
                           qx_data=qx_data, qy_data=qy_data,
                           q_data=q_data, mask=mask)

        clone._xaxis = self._xaxis
        clone._yaxis = self._yaxis
        clone._zaxis = self._zaxis
        clone._xunit = self._xunit
        clone._yunit = self._yunit
        clone._zunit = self._zunit
        clone.x_bins = self.x_bins
        clone.y_bins = self.y_bins

        clone.title = self.title
        clone.run = self.run
        clone.filename = self.filename
        clone.instrument = self.instrument
        clone.notes = deepcopy(self.notes)
        clone.process = deepcopy(self.process)
        clone.detector = deepcopy(self.detector)
        clone.sample = deepcopy(self.sample)
        clone.source = deepcopy(self.source)
        clone.collimation = deepcopy(self.collimation)
        clone.trans_spectrum = deepcopy(self.trans_spectrum)
        clone.meta_data = deepcopy(self.meta_data)
        clone.errors = deepcopy(self.errors)

        return clone

    def copy_from_datainfo(self, data2d):
        """
        copy value of Data2D of type DataLoader.data_info
        """
        self.data = copy.deepcopy(data2d.data)
        self.qx_data = copy.deepcopy(data2d.qx_data)
        self.qy_data = copy.deepcopy(data2d.qy_data)
        self.q_data = copy.deepcopy(data2d.q_data)
        self.mask = copy.deepcopy(data2d.mask)
        self.err_data = copy.deepcopy(data2d.err_data)
        self.x_bins = copy.deepcopy(data2d.x_bins)
        self.y_bins = copy.deepcopy(data2d.y_bins)
        if data2d.dqx_data is not None:
            self.dqx_data = copy.deepcopy(data2d.dqx_data)
        if data2d.dqy_data is not None:
            self.dqy_data = copy.deepcopy(data2d.dqy_data)
        self.xmin = data2d.xmin
        self.xmax = data2d.xmax
        self.ymin = data2d.ymin
        self.ymax = data2d.ymax
        if hasattr(data2d, "zmin"):
            self.zmin = data2d.zmin
        if hasattr(data2d, "zmax"):
            self.zmax = data2d.zmax
        self.xaxis(data2d._xaxis, data2d._xunit)
        self.yaxis(data2d._yaxis, data2d._yunit)
        self.title = data2d.title

    def _validity_check(self, other):
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
        err_other = None
        TOLERANCE = 0.01
        msg_base = "Incompatible data sets: q-values do not match: "
        if isinstance(other, Data2D):
            # Check that data lengths are the same
            if (len(self.data) != len(other.data)
                    or len(self.qx_data) != len(other.qx_data)
                    or len(self.qy_data) != len(other.qy_data)):
                msg = "Unable to perform operation: data length are not equal"
                raise ValueError(msg)
            for ind in range(len(self.data)):
                if (fabs(self.qx_data[ind] - other.qx_data[ind])
                        > fabs(self.qx_data[ind])*TOLERANCE):
                    msg = f"{msg_base}{self.qx_data[ind]} {other.qx_data[ind]}"
                    raise ValueError(msg)
                if (fabs(self.qy_data[ind] - other.qy_data[ind])
                        > fabs(self.qy_data[ind])*TOLERANCE):
                    msg = f"{msg_base}{self.qy_data[ind]} {other.qy_data[ind]}"
                    raise ValueError(msg)

            # Check that the scales match
            err_other = other.err_data
            if (other.err_data is None
                    or (len(other.err_data) != len(other.data))):
                err_other = np.zeros(len(other.data))

        # Check that we have errors, otherwise create zero vector
        err = self.err_data
        if self.err_data is None or (len(self.err_data) != len(self.data)):
            err = np.zeros(len(other.data))
        return err, err_other

    def _perform_operation(self, other, operation):
        """
        Perform 2D operations between data sets

        :param other: other data set
        :param operation: function defining the operation
        """
        # First, check the data compatibility
        dy, dy_other = self._validity_check(other)
        result = self.clone_without_data(np.size(self.data))
        if self.dqx_data is None or self.dqy_data is None:
            result.dqx_data = None
            result.dqy_data = None
        else:
            result.dqx_data = np.zeros(len(self.data))
            result.dqy_data = np.zeros(len(self.data))
        for i in range(np.size(self.data)):
            result.data[i] = self.data[i]
            if (self.err_data is not None
                    and np.size(self.data) == np.size(self.err_data)):
                result.err_data[i] = self.err_data[i]
            if self.dqx_data is not None:
                result.dqx_data[i] = self.dqx_data[i]
            if self.dqy_data is not None:
                result.dqy_data[i] = self.dqy_data[i]
            result.qx_data[i] = self.qx_data[i]
            result.qy_data[i] = self.qy_data[i]
            result.q_data[i] = self.q_data[i]
            result.mask[i] = self.mask[i]

            a = Uncertainty(self.data[i], dy[i]**2)
            if isinstance(other, Data2D):
                b = Uncertainty(other.data[i], dy_other[i]**2)
                if other.dqx_data is not None and result.dqx_data is not None:
                    result.dqx_data[i] *= self.dqx_data[i]
                    result.dqx_data[i] += (other.dqx_data[i]**2)
                    result.dqx_data[i] /= 2
                    result.dqx_data[i] = sqrt(result.dqx_data[i])
                if other.dqy_data is not None and result.dqy_data is not None:
                    result.dqy_data[i] *= self.dqy_data[i]
                    result.dqy_data[i] += (other.dqy_data[i]**2)
                    result.dqy_data[i] /= 2
                    result.dqy_data[i] = sqrt(result.dqy_data[i])
            else:
                b = other
            output = operation(a, b)
            result.data[i] = output.x
            result.err_data[i] = sqrt(fabs(output.variance))
        return result

    @staticmethod
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
        if not isinstance(other, Data2D):
            msg = "Unable to perform operation: different types of data set"
            raise ValueError(msg)
        return True

    def _perform_union(self, other):
        """
        Perform 2D operations between data sets

        :param other: other data set
        :param operation: function defining the operation
        """
        # First, check the data compatibility
        self._validity_check_union(other)
        result = self.clone_without_data(np.size(self.data)
                                         + np.size(other.data))
        result.xmin = self.xmin
        result.xmax = self.xmax
        result.ymin = self.ymin
        result.ymax = self.ymax
        if (self.dqx_data is None or self.dqy_data is None
                or other.dqx_data is None or other.dqy_data is None):
            result.dqx_data = None
            result.dqy_data = None
        else:
            result.dqx_data = np.zeros(len(self.data) + np.size(other.data))
            result.dqy_data = np.zeros(len(self.data) + np.size(other.data))

        result.data = np.append(self.data, other.data)
        result.qx_data = np.append(self.qx_data, other.qx_data)
        result.qy_data = np.append(self.qy_data, other.qy_data)
        result.q_data = np.append(self.q_data, other.q_data)
        result.mask = np.append(self.mask, other.mask)
        if result.err_data is not None:
            result.err_data = np.append(self.err_data, other.err_data)
        if self.dqx_data is not None:
            result.dqx_data = np.append(self.dqx_data, other.dqx_data)
        if self.dqy_data is not None:
            result.dqy_data = np.append(self.dqy_data, other.dqy_data)

        return result


def combine_data_info_with_plottable(data, datainfo):
    """
    A function that combines the DataInfo data in self.current_datainto with a
    plottable_1D or 2D data object.

    :param data: A plottable_1D or plottable_2D data object
    :param datainfo: A DataInfo object to be combined with the plottable
    :return: A fully specified Data1D or Data2D object
    """

    # TODO: Abstract this out -> everything is nox x, y, z
    if isinstance(data, Plottable1D):
        final_dataset = Data1D(data.x, data.y, isSesans=datainfo.isSesans)
        final_dataset.dx = data.dx
        final_dataset.dy = data.dy
        final_dataset.dxl = data.dxl
        final_dataset.dxw = data.dxw
        final_dataset.x_unit = data._xunit
        final_dataset.y_unit = data._yunit
        final_dataset.xaxis(data._xaxis, data._xunit)
        final_dataset.yaxis(data._yaxis, data._yunit)
    elif isinstance(data, Plottable2D):
        final_dataset = Data2D(data.data, data.err_data, data.qx_data,
                               data.qy_data, data.q_data, data.mask,
                               data.dqx_data, data.dqy_data)
        final_dataset.xaxis(data._xaxis, data._xunit)
        final_dataset.yaxis(data._yaxis, data._yunit)
        final_dataset.zaxis(data._zaxis, data._zunit)
        final_dataset.x_bins = data.x_bins
        final_dataset.y_bins = data.y_bins
    else:
        return_string = ("Should Never Happen: _combine_data_info_with_plottabl"
                         "e input is not a plottable1d or plottable2d data "
                         "object")
        return return_string

    if hasattr(data, "xmax"):
        final_dataset.xmax = data.xmax
    if hasattr(data, "ymax"):
        final_dataset.ymax = data.ymax
    if hasattr(data, "xmin"):
        final_dataset.xmin = data.xmin
    if hasattr(data, "ymin"):
        final_dataset.ymin = data.ymin
    final_dataset.isSesans = datainfo.isSesans
    final_dataset.title = datainfo.title
    final_dataset.run = datainfo.run
    final_dataset.run_name = datainfo.run_name
    final_dataset.filename = datainfo.filename
    final_dataset.notes = datainfo.notes
    final_dataset.process = datainfo.process
    final_dataset.instrument = datainfo.instrument
    final_dataset.detector = datainfo.detector
    final_dataset.sample = datainfo.sample
    final_dataset.source = datainfo.source
    final_dataset.collimation = datainfo.collimation
    final_dataset.trans_spectrum = datainfo.trans_spectrum
    final_dataset.meta_data = datainfo.meta_data
    final_dataset.errors = datainfo.errors
    return final_dataset
