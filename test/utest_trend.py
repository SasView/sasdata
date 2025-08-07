from os import listdir, path

import pytest

import sasdata.temp_ascii_reader as ascii_reader
from sasdata.ascii_reader_metadata import AsciiMetadataCategory
from sasdata.quantities.units import per_angstrom, per_nanometer
from sasdata.temp_ascii_reader import AsciiReaderParams
from sasdata.trend import Trend

mumag_test_directories = [
    'FeNiB_perpendicular_Bersweiler_et_al',
    'Nanoperm_perpendicular_Honecker_et_al',
    'NdFeB_parallel_Bick_et_al'
]

custom_test_directory = 'custom_test'

def get_files_to_load(directory_name: str) -> list[str]:
    load_from = path.join(path.dirname(__file__), 'trend_test_data', directory_name)
    base_filenames_to_load = listdir(load_from)
    files_to_load = [path.join(load_from, basename) for basename in base_filenames_to_load]
    return files_to_load

@pytest.mark.parametrize('directory_name', mumag_test_directories)
def test_trend_build_interpolate(directory_name: str):
    """Try to build a trend object on the MuMag datasets"""
    files_to_load = get_files_to_load(directory_name)
    params = AsciiReaderParams(
        filenames=files_to_load,
        columns=[('Q', per_nanometer), ('I', per_nanometer), ('dI', per_nanometer)],
    )
    params.separator_dict['Whitespace'] = True
    params.metadata.master_metadata['magnetic'] = AsciiMetadataCategory(
        values={
            'counting_index': 0,
            'applied_magnetic_field': 1,
            'saturation_magnetization': 2,
            'demagnetizing_field': 3
        }
    )
    data = ascii_reader.load_data(params)
    trend = Trend(
        data=data,
        trend_axis=['magnetic', 'applied_magnetic_field']
    )
    # Initially, the q axes in this date don't exactly match
    to_interpolate_on = 'Q'
    assert not trend.all_axis_match(to_interpolate_on)
    interpolated_trend = trend.interpolate(to_interpolate_on)
    assert interpolated_trend.all_axis_match(to_interpolate_on)

def test_trend_q_axis_match():
    files_to_load = get_files_to_load(custom_test_directory)
    params = AsciiReaderParams(
        filenames=files_to_load,
        columns=[('Q', per_angstrom), ('I', per_angstrom)]
    )
    params.metadata.master_metadata['magnetic'] = AsciiMetadataCategory(
        values={
            'counting_index': 0,
        }
    )
    data = ascii_reader.load_data(params)
    trend = Trend(
        data=data,
        trend_axis=['magnetic', 'counting_index']
    )
    assert trend.all_axis_match('Q')
