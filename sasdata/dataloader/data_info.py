"""
    @Deprecation: This module contains placeholders for deprecated data objects. All deprecated objects return
"""

from typing import Iterable, Optional

import sasdata.data.meta_data as meta_data
import sasdata.data.data_info as data_info
import sasdata.data.data as data_new
import sasdata.data.plottables as plottables
from sasdata.data_util.deprecation import deprecated


NEUTRON = meta_data.NEUTRON
XRAY = meta_data.XRAY
MUON = meta_data.MUON
ELECTRON = meta_data.ELECTRON


class plottable_1D(data_new.Plottable):
    """@Deprecated: Superseded by a number of 1-dimensional sasdata.data.data.Plottable classes"""
    @deprecated(replaced_with='sasdata.data.data.Plottable')
    def __new__(cls, x: Iterable, y: Iterable, dx: Optional[Iterable] = None, dy: Optional[Iterable] = None,
                 dxl: Optional[Iterable] = None, dxw: Optional[Iterable] = None, lam: Optional[Iterable] = None,
                 dlam: Optional[Iterable] = None, mask: Optional[Iterable] = None):
        if lam is not None and dlam is not None:
            return plottables.SpinEchoSANS(x, y, dx, dy, mask, lam, dlam)
        elif dxl is not None or dxw is not None:
            return plottables.SlitSmeared1D(x, y, dx, dy, mask, dxl, dxw)
        else:
            return plottables.Plottable1D(x, y, dx, dy, mask)


class plottable_2D(data_new.Plottable2D):
    """@Deprecated: Superseded by a number of 2-dimensional sasdata.data.data.Plottable classes"""
    @deprecated(replaced_with='sasdata.data.data.Plottable2D')
    def __init__(self, x: Iterable, y: Iterable, z: Iterable, dx: Optional[Iterable] = None,
                 dy: Optional[Iterable] = None, dz: Optional[Iterable] = None, mask: Optional[Iterable] = None):
        super(plottable_2D, self).__init__(x, y, z, dx, dy, dz, mask)


class Vector(meta_data.Vector):
    """@Deprecated: Superseded by the sasdata.data.meta_data.Vector class"""
    @deprecated(replaced_with='sasdata.data.meta_data.Vector')
    def __init__(self, x, y, z):
        super(Vector, self).__init__(x, y, z)


class Detector(meta_data.Detector):
    """@Deprecated: Superseded by the sasdata.data.meta_data.Detector class"""
    @deprecated(replaced_with='sasdata.data.meta_data.Detector')
    def __init__(self):
        super().__init__()


class Aperture(meta_data.Aperture):
    """@Deprecated: Superseded by the sasdata.data.meta_data.Aperture class"""
    @deprecated(replaced_with='sasdata.data.meta_data.Aperture')
    def __init__(self):
        super().__init__()


class Collimation(meta_data.Collimation):
    """@Deprecated: Superseded by the sasdata.data.meta_data.Collimation class"""
    @deprecated(replaced_with='sasdata.data.meta_data.Collimation')
    def __init__(self):
        super().__init__()


class Source(meta_data.Source):
    """@Deprecated: Superseded by the sasdata.data.meta_data.Source class"""
    @deprecated(replaced_with='sasdata.data.meta_data.Source')
    def __init__(self):
        super().__init__()


class Sample(meta_data.Sample):
    """@Deprecated: Superseded by the sasdata.data.meta_data.Sample class"""
    @deprecated(replaced_with='sasdata.data.meta_data.Sample')
    def __init__(self):
        super().__init__()


class Process(meta_data.Process):
    """@Deprecated: Superseded by the sasdata.data.meta_data.Process class"""
    @deprecated(replaced_with='sasdata.data.meta_data.Process')
    def __init__(self):
        super().__init__()


class TransmissionSpectrum(meta_data.TransmissionSpectrum):
    """@Deprecated: Superseded by the sasdata.data.meta_data.TransmissionSpectrum class"""
    @deprecated(replaced_with='sasdata.data.meta_data.TransmissionSpectrum')
    def __init__(self):
        super().__init__()


class DataInfo(data_info.DataInfo):
    """@Deprecated: Superseded by the sasdata.data.data_info.DataInfo class"""
    @deprecated(replaced_with='sasdata.data.data_info.DataInfo')
    def __init__(self):
        super().__init__()


class Data1D(data_new.Data1D):
    """@Deprecated: Superseded by the sasdata.data.data.Data class"""
    @deprecated(replaced_with='sasdata.data.data.Data')
    def __init__(self, x: Iterable, y: Iterable, dx: Optional[Iterable] = None, dy: Optional[Iterable] = None,
                 lam: Optional[Iterable] = None, dlam: Optional[Iterable] = None, isSesans: Optional[bool] = False):
        # TODO: This only returns a Data1D object -> create single Data class that inherits from any plottable type
        super().__init__(x, y, dx, dy, None)


class Data2D(data_new.Data2D):
    """@Deprecated: Superseded by the sasdata.data.data.Data class"""
    @deprecated(replaced_with='sasdata.data.data.Data')
    def __init__(self, data: Iterable, err_data: Optional[Iterable] = None, qx_data: Optional[Iterable] = None,
                 qy_data: Optional[Iterable] = None, q_data: Optional[Iterable] = None, mask: Optional[Iterable] = None,
                 dqx_data: Optional[Iterable] = None, dqy_data: Optional[Iterable] = None, xmin: Optional[int] = None,
                 xmax: Optional[int] = None, ymin: Optional[int] = None, ymax: Optional[int] = None,
                 zmin: Optional[int] = None, zmax: Optional[int] = None):
        # TODO: This only returns a Data2D object -> create single Data class that inherits from any plottable type
        super().__init__(data, err_data, qx_data, qy_data, q_data, mask, dqx_data, dqy_data)


@deprecated(replaced_with='sasdata.data.data.combine_data_info_with_plottable')
def combine_data_info_with_plottable(plottable: data_new.Plottable, datainfo: data_info.DataInfo) -> data_new.Data:
    """
    @Deprecated: Superseded by the sasdata.data.data.combine_data_info_with_plottable function

    A function that combines the DataInfo data in self.current_datainto with a
    plottable_1D or 2D data object.

    :param data: Any Plottable data object
    :param datainfo: A DataInfo object to be combined with the plottable
    :return: A fully specified Data object
    """
    return data_new.combine_data_info_with_plottable(plottable, datainfo)
