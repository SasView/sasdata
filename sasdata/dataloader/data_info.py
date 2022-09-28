# TODO: Add documentation to say this entire file is deprecated

"""
    Module that contains placeholders for old version of the data objects.

    A good description of the data members can be found in
    the CanSAS 1D XML data format:

    http://www.smallangles.net/wgwiki/index.php/cansas1d_documentation
"""

import sasdata.data.meta_data as meta_data
import sasdata.data.data_info as data_info
import sasdata.data.data as data
import sasdata.data.plottables as plottables
from sasdata.data_util.deprecation import deprecated


NEUTRON = meta_data.NEUTRON
XRAY = meta_data.XRAY
MUON = meta_data.MUON
ELECTRON = meta_data.ELECTRON


@deprecated(replaced_with="sasdata.data.meta_data.Vector()")
def Vector(x, y, z):
    return meta_data.Vector(x, y, z)


@deprecated(replaced_with="sasdata.data.meta_data.Detector()")
def Detector():
    return meta_data.Detector()


@deprecated(replaced_with="sasdata.data.meta_data.Aperture()")
def Aperture():
    return meta_data.Aperture()


@deprecated(replaced_with="sasdata.data.meta_data.Collimation()")
def Collimation():
    return meta_data.Collimation()


@deprecated(replaced_with="sasdata.data.meta_data.Source()")
def Source():
    return meta_data.Source()


@deprecated(replaced_with="sasdata.data.meta_data.Sample()")
def Sample():
    return meta_data.Sample()


@deprecated(replaced_with="sasdata.data.meta_data.Process()")
def Process():
    return meta_data.Process()


@deprecated(replaced_with="sasdata.data.meta_data.TransmissionSpectrum()")
def TransmissionSpectrum():
    return meta_data.TransmissionSpectrum()


@deprecated(replaced_with="sasdata.data.data_info.DataInfo()")
def DataInfo():
    return data_info.DataInfo()


@deprecated(replaced_with="sasdata.data.data.Data1D()")
def Data1D(x=None, y=None, dx=None, dy=None, lam=None, dlam=None, isSesans=False):
    # TODO: create specific objects based on inputs
    mask = None
    if isSesans:
        plottables.SpinEchoSANS(x, y, dx, dy, mask, lam, dlam)
    return data.Data1D(x, y, dx, dy, mask)


@deprecated(replaced_with="sasdata.data.data.Data2D()")
def Data2D(data=None, err_data=None, qx_data=None,
                 qy_data=None, q_data=None, mask=None,
                 dqx_data=None, dqy_data=None,
                 xmin=None, xmax=None, ymin=None, ymax=None,
                 zmin=None, zmax=None):
    # TODO: implement usage of remaining values
    return data.Data2D(x=qx_data, y=qy_data, z=data, dx=dqx_data, dy=dqy_data, dz=err_data, mask=mask)


@deprecated(replaced_with="sasdata.data.data.combine_data_info_with_plottable()")
def combine_data_info_with_plottable(data, datainfo):
    """
    A function that combines the DataInfo data in self.current_datainto with a
    plottable_1D or 2D data object.

    :param data: A plottable_1D or plottable_2D data object
    :param datainfo: A DataInfo object to be combined with the plottable
    :return: A fully specified Data1D or Data2D object
    """

    return data.combine_data_info_with_plottable(data, datainfo)
