import pytest
from os import path, listdir
from sasdata.ascii_reader_metadata import AsciiMetadataCategory, AsciiReaderMetadata
from sasdata.quantities.units import per_nanometer, per_angstrom
from sasdata.temp_ascii_reader import AsciiReaderParams
import sasdata.temp_ascii_reader as ascii_reader
from sasdata.trend import Trend

mumag_test_directories = [
    'FeNiB_perpendicular_Bersweiler_et_al',
    'Nanoperm_perpendicular_Honecker_et_al',
    'NdFeB_parallel_Bick_et_al'
]

custom_test_directory = 'custom_test'

@pytest.mark.parametrize('directory_name', mumag_test_directories)
def test_trend_build(directory_name: str):
    """Try to build a trend object on the MuMag datasets"""
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
    for basename in base_filenames_to_load:
        metadata.filename_separator[basename] = '_'
        metadata.filename_specific_metadata[basename] = {}

    params = AsciiReaderParams(
        filenames=files_to_load,
        starting_line=0,
        columns=[('Q', per_nanometer), ('I', per_nanometer), ('dI', per_nanometer)],
        excluded_lines=set(),
        separator_dict={'Whitespace': True, 'Comma': False, 'Tab': False},
        metadata=metadata
    )
    data = ascii_reader.load_data(params)
    trend = Trend(
        data=data,
        trend_axis=['magnetic', 'applied_magnetic_field']
    )
    # TODO: Trend setup without error but should have some verificaton that it works.

# TODO: Some of this loading logic is repeated. Can it be abstracted into its own function?
def test_trend_q_axis_match():
    load_from = path.join(path.dirname(__file__), 'trend_test_data', custom_test_directory)
    base_filenames_to_load = listdir(load_from)
    files_to_load = [path.join(load_from, basename) for basename in base_filenames_to_load]
    metadata = AsciiReaderMetadata()
    metadata.master_metadata['magnetic'] = AsciiMetadataCategory(
        values={
            'counting_index': 0,
        }
    )
    for basename in base_filenames_to_load:
        metadata.filename_separator[basename] = '_'
        metadata.filename_specific_metadata[basename] = {}

    params = AsciiReaderParams(
        filenames=files_to_load,
        starting_line=0,
        columns=[('Q', per_angstrom), ('I', per_angstrom)],
        excluded_lines=set(),
        separator_dict={'Whitespace': False, 'Comma': True, 'Tab': False},
        metadata=metadata
    )
    data = ascii_reader.load_data(params)
    trend = Trend(
        data=data,
        trend_axis=['counting_index']
    )
