from dataclasses import dataclass
from enum import Enum
from typing import Self, TypeVar

from sasdata.quantities.quantity import NamedQuantity

DataType = TypeVar("DataType")

""" Sasdata metadata tree """

def shorten_string(string):
    lines = string.split("\n")
    if len(lines) <= 1:
        return string
    else:
        return lines[0][:30] + " ... " + lines[-1][-30:]

@dataclass
class Dataset[DataType]:
    name: str
    data: DataType
    attributes: dict[str, Self | str]

    def summary(self, indent_amount: int = 0, indent: str = "  ") -> str:

        s = f"{indent*indent_amount}{self.name.split("/")[-1]}:\n"
        s += f"{indent*(indent_amount+1)}{shorten_string(str(self.data))}\n"
        for key in self.attributes:
            value = self.attributes[key]
            if isinstance(value, (Group, Dataset)):
                value_string = value.summary(indent_amount+1, indent)
            else:
                value_string = f"{indent * (indent_amount+1)}{key}: {shorten_string(repr(value))}\n"

            s += value_string

        return s

@dataclass
class Group:
    name: str
    children: dict[str, Self | Dataset]

    def summary(self, indent_amount: int=0, indent="  "):
        s = f"{indent*indent_amount}{self.name.split("/")[-1]}:\n"
        for key in self.children:
            s += self.children[key].summary(indent_amount+1, indent)

        return s

class Function:
    """ Representation of a (data driven) function, such as I vs Q """

    def __init__(self, abscissae: list[NamedQuantity], ordinate: NamedQuantity):
        self.abscissae = abscissae
        self.ordinate = ordinate


class FunctionType(Enum):
    """ What kind of function is this, should not be relied upon to be perfectly descriptive

    The functions might be parametrised by more variables than the specification
    """
    UNKNOWN = 0
    SCATTERING_INTENSITY_VS_Q = 1
    SCATTERING_INTENSITY_VS_Q_2D = 2
    SCATTERING_INTENSITY_VS_Q_3D = 3
    SCATTERING_INTENSITY_VS_ANGLE = 4
    UNKNOWN_METADATA = 20
    TRANSMISSION = 21
    POLARISATION_EFFICIENCY = 22
    UNKNOWN_REALSPACE = 30
    SESANS = 31
    CORRELATION_FUNCTION_1D = 32
    CORRELATION_FUNCTION_2D = 33
    CORRELATION_FUNCTION_3D = 34
    INTERFACE_DISTRIBUTION_FUNCTION = 35
    PROBABILITY_DISTRIBUTION = 40
    PROBABILITY_DENSITY = 41

def function_type_identification_key(names):
    """ Create a key from the names of data objects that can be used to assign a function type"""
    return ":".join([s.lower() for s in sorted(names)])

function_fields_to_type = [
    (["Q"], "I", FunctionType.SCATTERING_INTENSITY_VS_Q),
    (["Qx", "Qy"], "I", FunctionType.SCATTERING_INTENSITY_VS_Q_2D),
    (["Qx", "Qy", "Qz"], "I", FunctionType.SCATTERING_INTENSITY_VS_Q_3D),
    (["Z"], "G", FunctionType.SESANS),
    (["lambda"], "T", FunctionType.TRANSMISSION)
]

function_fields_lookup = {
    function_type_identification_key(inputs + [output]): function_type for inputs, output, function_type in function_fields_to_type
}

def build_main_data(data: list[NamedQuantity]) -> Function:
    names = [datum.name for datum in data]
    identifier = function_type_identification_key(names)

    if identifier in function_fields_lookup:
        function_type = function_fields_lookup[identifier]
    else:
        function_type = FunctionType.UNKNOWN

    match function_type:
        case FunctionType.UNKNOWN:
            pass
        case _:
            raise NotImplementedError("Unknown ")

def key_tree(data: Group | Dataset, indent_amount=0, indent: str = "  ") -> str:
    """ Show a metadata tree, showing the names of they keys used to access them"""
    s = ""
    if isinstance(data, Group):
        for key in data.children:
            s += indent*indent_amount + key + "\n"
            s += key_tree(data.children[key], indent_amount=indent_amount+1, indent=indent)

    if isinstance(data, Dataset):
        s += indent*indent_amount + "[data]\n"
        for key in data.attributes:
            s += indent*indent_amount + key + "\n"
            s += key_tree(data.attributes[key], indent_amount=indent_amount+1, indent=indent)

    return s