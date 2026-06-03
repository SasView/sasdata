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
    """Calculates and appends Qz to SasData if Qx, Qy, and wavelength are all present"""
    # if Qz is not already in the dataset, but Qx and Qy are
    if 'Qz' not in data._data_contents and 'Qx' in data._data_contents and 'Qy' in data._data_contents:
        # we start by making the approximation that qz=0
        data._data_contents['Qz'] = 0*data._data_contents['Qx']

        # now check if metadata has wavelength information
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

        if wavelength is not None:
            # we can deduce the value of qz from qx and qy
            # if we have the wavelength
            qx = data._data_contents['Qx']
            qy = data._data_contents['Qy']

            # this is how you convert qx, qy, and wavelength to qz
            k0 = 2*np.pi/wavelength
            qz = k0-(k0**2-qx**2-qy**2)**(0.5)

            data._data_contents['Qz'] = qz

