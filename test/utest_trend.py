import pytest
from os import path, listdir
from sasdata.ascii_reader_metadata import AsciiMetadataCategory, AsciiReaderMetadata
from sasdata.quantities.units import per_nanometer
from sasdata.temp_ascii_reader import AsciiReaderParams
import sasdata.temp_ascii_reader as ascii_reader
from sasdata.trend import Trend

test_directories = [
    'FeNiB_perpendicular_Bersweiler_et_al',
    'Nanoperm_perpendicular_Honecker_et_al',
    'NdFeB_parallel_Bick_et_al'
]

@pytest.mark.parametrize('directory_name', test_directories)
def test_trend_build(directory_name: str):
    """Try to build a trend object on the MuMag datasets, and see if all the Q items match (as they should)."""
    load_from = path.join(path.dirname(__file__), 'trend_test_data', directory_name)
    base_filenames_to_load = listdir(load_from)
    files_to_load = [path.join(load_from, basename) for basename in base_filenames_to_load]

    metadata = AsciiReaderMetadata()
    metadata.master_metadata['magnetic'] = AsciiMetadataCategory(
        values={
            'counting_index': 0,
            'applied_magnetic_field': 1,
            'saturation_magnetization': 2,
            'demagnetizing_field': 3
        }
    )

    params = AsciiReaderParams(
        filenames=files_to_load,
        starting_line=0,
        columns=[('Q', per_nanometer), ('I', per_nanometer), ('dI', per_nanometer)],
        excluded_lines=set(),
        separator_dict={'Whitespace': True, 'Comma': False, 'Tab': False},
        metadata=metadata,
    )
    data = ascii_reader.load_data(params)
    trend = Trend(
        data=data,
        trend_axis=['magnetic', 'applied_magnetic_field']
    )
    assert trend.all_axis_match('Q')
