from typing import Generic, TypeVar

from numpy._typing import ArrayLike

from sasdata.quantities.quantities import Unit, Quantity

    def __init__(self, target_object: AccessorTarget):

        # Name of the instrument [string]
        self.name = StringAccessor(target_object, "name")

        # Sample to detector distance [float] [mm]
        self.distance = LengthAccessor[float](target_object,
                                              "distance",
                                              "distance.units",
                                              default_unit=units.millimeters)

        # Offset of this detector position in X, Y,
        # (and Z if necessary) [Vector] [mm]
        self.offset = LengthAccessor[ArrayLike](target_object,
                                                "offset",
                                                "offset.units",
                                                default_unit=units.millimeters)

        self.orientation = AngleAccessor[ArrayLike](target_object,
                                                    "orientation",
                                                    "orientation.units",
                                                    default_unit=units.degrees)

        self.beam_center = LengthAccessor[ArrayLike](target_object,
                                                     "beam_center",
                                                     "beam_center.units",
                                                     default_unit=units.millimeters)

        # Pixel size in X, Y, (and Z if necessary) [Vector] [mm]
        self.pixel_size = LengthAccessor[ArrayLike](target_object,
                                                    "pixel_size",
                                                    "pixel_size.units",
                                                    default_unit=units.millimeters)

        # Slit length of the instrument for this detector.[float] [mm]
        self.slit_length = LengthAccessor[float](target_object,
                                                 "slit_length",
                                                 "slit_length.units",
                                                 default_unit=units.millimeters)

class RawMetaData:
    pass

class MetaData:
    pass


FieldDataType = TypeVar("FieldDataType")
OutputDataType = TypeVar("OutputDataType")

class Accessor(Generic[FieldDataType, OutputDataType]):
    def __init__(self, target_field: str):
        self._target_field = target_field

    def _raw_values(self) -> FieldDataType:
        raise NotImplementedError("not implemented in base class")

    @property
    def value(self) -> OutputDataType:
        raise NotImplementedError("value not implemented in base class")



class QuantityAccessor(Accessor[ArrayLike, Quantity[ArrayLike]]):
    def __init__(self, target_field: str, units_field: str | None = None):
        super().__init__(target_field)
        self._units_field = units_field

    def _get_units(self) -> Unit:
        pass

    def _raw_values(self) -> ArrayLike:
        pass


class StringAccessor(Accessor[str]):
    @property
    def value(self) -> str:
        return self._raw_values()


class LengthAccessor(QuantityAccessor):
    @property
    def m(self):
        return self.value.in_units_of("m")


class TimeAccessor(QuantityAccessor):
    pass


class TemperatureAccessor(QuantityAccessor):
    pass


class AbsoluteTemperatureAccessor(QuantityAccessor):
    pass

