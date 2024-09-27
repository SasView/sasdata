from typing import TypeVar, Any, Self
from dataclasses import dataclass

from quantities.quantity import NamedQuantity

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
    attributes: dict[str, Self]

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

@dataclass
class RawData:
    name: str
    data_contents: list[NamedQuantity]
    raw_metadata: dict[str, Dataset | Group]

    def __repr__(self):
        indent = "  "

        s = f"{self.name}\n"
        for key in self.raw_metadata:
            s += self.raw_metadata[key].summary(1, indent)

        return s