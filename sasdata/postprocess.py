"""

Post processing for loaded files

"""

import numpy as np

from sasdata.data import SasData

def fix_mantid_units_error(data: SasData) -> SasData:
    pass



def apply_fixes(data: SasData, mantid_unit_error=True):
    if mantid_unit_error:
        data = fix_mantid_units_error(data)

    return data


def deduce_qz(data: SasData):
    # check if metadata has wavelength information and data is 2D
    wavelength = getattr(
        getattr(
            getattr(
                getattr(data, "metadata", None),
                "instrument",
                None
            ),
            "source",
            None
        ),
        "wavelength",
        None
    )
    # we start by making the approximation that qz=0
    data._data_contents['Qz'] = 0*data._data_contents['Qx']
    
    if wavelength is not None:
        # we can deduce the value of qz from qx and qy
        qx = data._data_contents['Qx'].value
        qy = data._data_contents['Qy'].value

        # this is how you convert qx, qy, and wavelength to qz
        k0 = 2*np.pi/wavelength
        twotheta = np.arcsin((qx**2 + qy**2) / k0)
        qz = (1 - np.cos(twotheta)) * k0

        data._data_contents['Qz'].value = qz

    return data