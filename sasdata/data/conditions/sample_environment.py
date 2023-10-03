"""Various classes that define sample environment (SE) conditions."""

from sasdata.data.condition import Condition, value_type, value_type_optional, str_type_optional
from sasdata.data_util.nxsunit import DIMENSIONS


class SampleEnvironment(Condition):
    _can_sas_class = "NXenvironment"

    def __init__(self, value: value_type, uncertainty: value_type_optional = None, units: str_type_optional = None):
        super().__init__(value, uncertainty, units)
        self._are_units_sensible(units)


class Temperature(SampleEnvironment):
    _can_sas_class = "NXtemperature"
    _name = "temperature"

    def _are_units_sensible(self, units: str):
        sensible_units = DIMENSIONS.get('temperature', {})
        if units not in sensible_units.keys():
            raise ValueError(f"The units, {units}, sent to {self.__class__} are not sensible for {self._name}.")
        super()._are_units_sensible(units)


class MagneticField(SampleEnvironment):
    _can_sas_class = "NXmagnet"
    _name = "magnetic field"

    def _are_units_sensible(self, units: str):
        sensible_units = DIMENSIONS.get('magnetism', {})
        if units not in sensible_units.keys():
            raise ValueError(f"The units, {units}, sent to {self.__class__} are not sensible for {self._name}.")
        super()._are_units_sensible(units)


class ShearRate(SampleEnvironment):
    _name = "shear rate"


class Pressure(SampleEnvironment):
    _name = "pressure"
