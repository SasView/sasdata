""" Information used for providing guesses about what text based files contain """

from dataclasses import dataclass

import sasdata.quantities.units as units

#
#   VERY ROUGH DRAFT - FOR PROTOTYPING PURPOSES
#

@dataclass
class DatasetType:
    name: str
    required: list[str]
    optional: list[str]
    expected_orders: list[list[str]]


one_dim = DatasetType(
            name="1D I vs Q",
            required=["Q", "I"],
            optional=["dI", "dQ", "Shadowfactor", "Qmean", "dQl", "dQw"],
            expected_orders=[
                ["Q", "I", "dI"],
                ["Q", "dQ", "I", "dI"]])

two_dim = DatasetType(
            name="2D I vs Q",
            required=["Qx", "Qy", "I"],
            optional=["dQx", "dQy", "dI", "Qz", "ShadowFactor", "mask"],
            expected_orders=[
                ["Qx", "Qy", "I"],
                ["Qx", "Qy", "I", "dI"],
                ["Qx", "Qy", "dQx", "dQy", "I", "dI"]])

sesans = DatasetType(
    name="SESANS",
    required=["SpinEchoLength", "Depolarisation", "Wavelength"],
    optional=["Transmission", "Polarisation"],
    expected_orders=[["z", "G"]])

dataset_types = {dataset.name for dataset in [one_dim, two_dim, sesans]}


#
# Some default units, this is not how they should be represented, some might not be correct
#
# The unit options should only be those compatible with the field
#

unit_kinds = {
    "Q": units.inverse_length,
    "I": units.inverse_length,
    "Qx": units.inverse_length,
    "Qy": units.inverse_length,
    "Qz": units.inverse_length,
    "dI": units.inverse_length,
    "dQ": units.inverse_length,
    "dQx": units.inverse_length,
    "dQy": units.inverse_length,
    "dQz": units.inverse_length,
    "SpinEchoLength": units.length,
    "Depolarisation": units.inverse_volume,
    "Wavelength": units.length,
    "Transmission": units.dimensionless,
    "Polarisation": units.dimensionless,
    "shadow": units.dimensionless,
    "temperature": units.temperature,
    "magnetic field": units.magnetic_flux_density
}

#
# Other possible fields. Ultimately, these should come out of the metadata structure
#

metadata_fields = [
    "temperature",
    "magnetic field",
]



