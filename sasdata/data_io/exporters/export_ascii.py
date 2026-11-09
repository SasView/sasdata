"""
Export SasData to delimited text files: whitespace separated multi-column ASCII files and comma separated (CSV) files.

The files start with comment lines describing the data, followed by a line naming each column and its units, e.g.
``Q(Ang^-1) I(cm^-1) dI(cm^-1) dQ(Ang^-1)``, so :class:`sasdata.data_io.importers.import_ascii.AsciiImporter` reads the
columns back with the correct names and units.

The columns are written in the order SasView expects, in the units SasView assumes (1/Å and 1/cm):

* 1D data: ``Q I dI dQ``, then any of ``dQw dQl Qmean Shadowfactor``. SasView reads these from ``.txt`` and ``.csv``
  files.
* 2D data: ``Qx Qy I dI Qz dQx dQy ShadowFactor``, then any of ``dQz mask``, using the NIST/IGOR 2D header that SasView
  reads from ``.dat`` files.

Columns that are needed to keep the positions of later columns, but have no data, are written as zeros and named
``<ignore>`` so they are skipped when the file is imported.
"""

from pathlib import Path
from typing import ClassVar

import numpy as np

from sasdata.data import SasMeasurement
from sasdata.data_io.exporter import Exporter
from sasdata.dataset_types import one_dim, three_dim, two_dim
from sasdata.quantities.units import Unit

IGNORE = "<ignore>"

# Columns written for each dataset type. The names are those of the quantities in the data, apart from errors, which
# are written as 'd' + the name of the quantity. The first group are the columns SasView reads by position, and so
# must be written in this order; the second group are optional extras, written after them when present.
COLUMN_ORDERS: dict[str, tuple[list[str], list[str]]] = {
    one_dim.name: (["Q", "I", "dI", "dQ"], ["dQw", "dQl", "Qmean", "Shadowfactor"]),
    two_dim.name: (["Qx", "Qy", "I", "dI", "Qz", "dQx", "dQy", "ShadowFactor"], ["dQz", "mask"]),
    three_dim.name: (["Qx", "Qy", "I", "dI", "Qz", "dQx", "dQy", "ShadowFactor"], ["dQz", "mask"]),
}

# The columns that must always be written, even if they are placeholders, for SasView to read the file correctly.
REQUIRED_COLUMNS = {
    one_dim.name: 2,
    two_dim.name: 3,
    three_dim.name: 5,
}

def format_value(value) -> str:
    return repr(float(value))


class DelimitedTextExporter(Exporter, register=False):
    """Shared implementation of the ASCII and CSV exporters. Not registered as an exporter itself."""

    # The string placed between values on each line.
    delimiter: ClassVar[str] = " "
    multiple_datasets = False

    def write(self, data: dict[str, SasMeasurement], path: Path):
        datum = next(iter(data.values()))
        names, columns = self.build_columns(datum)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for line in self.header_lines(datum, names):
                f.write(line + "\n")
            for row in zip(*columns):
                f.write(self.delimiter.join(format_value(v) for v in row) + "\n")

    def header_lines(self, datum: SasMeasurement, names: list[str]) -> list[str]:
        """Comment lines describing the data, finishing with the line naming the columns."""
        metadata = datum.metadata
        lines = [
            f"# {datum.name}",
            f"# Exported by sasdata as {datum.dataset_type.name} data",
        ]
        if metadata.title:
            lines.append(f"# Title: {metadata.title}")
        if metadata.run:
            lines.append(f"# Run: {' '.join(metadata.run)}")
        lines.append(self.delimiter.join(names))
        return lines

    def build_columns(self, datum: SasMeasurement) -> tuple[list[str], list[np.ndarray]]:
        """
        Find the columns to write. Returns the column headings (name and units) and the values of each column, in the
        units given in the headings.
        """
        contents = {name: quantity for name, quantity in datum._data_contents.items()}
        positional, extras = COLUMN_ORDERS[datum.dataset_type.name]
        available: dict[str, tuple[np.ndarray, Unit]] = {}
        for name, quantity in contents.items():
            available[name] = (np.asarray(quantity.value, dtype=float), quantity.units)
            if quantity.has_error and f"d{name}" not in contents:
                available[f"d{name}"] = (np.asarray(quantity.standard_error.value, dtype=float), quantity.units)

        # Write the positional columns up to the last one with data, filling any gaps with placeholders.
        present = [index for index, name in enumerate(positional) if name in available]
        last = max(present + [REQUIRED_COLUMNS[datum.dataset_type.name] - 1])
        extra_names = [name for name in extras if name in available]
        if extra_names:
            last = len(positional) - 1
        size = np.asarray(contents["I"].value).size

        headings: list[str] = []
        columns: list[np.ndarray] = []
        for name in positional[:last + 1] + extra_names:
            if name in available:
                values, unit = available[name]
                headings.append(f"{name}({unit.ascii_symbol})")
                columns.append(np.asarray(values, dtype=float).reshape(-1))
            else:
                headings.append(IGNORE)
                columns.append(np.zeros(size))
        return headings, columns


class AsciiExporter(DelimitedTextExporter):
    """
    Exporter for whitespace separated multi-column ASCII files. Supports 1D and 2D data, one dataset per file.

    SasView chooses how to read text files from their extension, so 1D data should be saved as ``.txt`` and 2D data as
    ``.dat`` for SasView to read it back. The sasdata importer reads either from the column headings.
    """

    name = "ASCII"
    aliases = ("text", "multi-column ascii", "txt", "dat")
    extensions = (".txt", ".dat", ".abs")
    supported_dataset_types = (one_dim, two_dim, three_dim)
    delimiter = " "

    def header_lines(self, datum: SasMeasurement, names: list[str]) -> list[str]:
        lines = super().header_lines(datum, names)
        if datum.dataset_type != one_dim:
            # The NIST/IGOR 2D header used by SasView's 2D ASCII reader, which needs to know whether the fourth column
            # is the uncertainty in I, and that the data starts on the line after 'ASCII data'.
            descriptions = {"Qx": "Qx", "Qy": "Qy", "I": "I(Qx,Qy)", "dI": "err(I)", "Qz": "Qz",
                            "dQx": "SigmaQ_parall", "dQy": "SigmaQ_perp", "ShadowFactor": "fSubS(beam stop shadow)"}
            column_names = [n.split("(")[0] for n in names]
            if column_names[3:4] == [IGNORE]:
                # No uncertainties: describe the column as something other than err(I).
                descriptions["dI"] = "unused"
            described = [descriptions.get(n, n) for n in column_names]
            lines.insert(-1, "Data columns are " + " - ".join(described))
            lines.append("ASCII data")
        return lines


class CsvExporter(DelimitedTextExporter):
    """Exporter for comma separated (CSV) files. Supports 1D data, one dataset per file."""

    name = "CSV"
    aliases = ("comma separated values", "comma-separated values")
    extensions = (".csv",)
    supported_dataset_types = (one_dim,)
    delimiter = ","
