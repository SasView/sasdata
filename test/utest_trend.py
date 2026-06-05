from os import listdir, path
import numpy as np

import pytest

import sasdata.temp_ascii_reader as ascii_reader
from sasdata.ascii_reader_metadata import AsciiMetadataCategory
from sasdata.quantities.units import per_angstrom, per_nanometer
from sasdata.temp_ascii_reader import AsciiReaderParams
from sasdata.temp_xml_reader import load_data as xml_load_data
from sasdata.temp_hdf5_reader import load_data as hdf_load_data
from sasdata.trend import Trend

mumag_test_directories = [
    'FeNiB_perpendicular_Bersweiler_et_al',
    'Nanoperm_perpendicular_Honecker_et_al',
    'NdFeB_parallel_Bick_et_al'
]

xml_file = path.join(path.dirname(__file__), 'trend_test_data', 'xml_test_files', 'cansas1d_notitle.xml')

hdf_file = path.join(path.dirname(__file__), 'trend_test_data', 'hdf_test_files', 'nxcansas_1Dand2D_multisasentry.h5')


custom_test_directory = 'custom_test'

def get_files_to_load(directory_name: str) -> list[str]:
    load_from = path.join(path.dirname(__file__), 'trend_test_data', directory_name)
    base_filenames_to_load = listdir(load_from)
    files_to_load = [path.join(load_from, basename) for basename in base_filenames_to_load]
    return files_to_load

def test_trend_build_from_xml():
    """
    Try to build a trend object from an XML file.
    The loader returns a dict of SasData objects (here only one, named 'SasData01')
    and currently a single Metadata (no tree of MetaNodes), but this should be
    corrected in the future in the XML reader, I guess!
    """
    data = xml_load_data(xml_file) # dict of SasData objects
    trend = Trend(
        data=[data['SasData01']],
        trend_axes={'entry': ['SASentry']}
    )
    assert (not trend.is_manual_axis('entry'))
    assert (len(trend.get_trend_values('entry')) == 1)

def test_trend_build_from_hdf5_with_multiple_axes():
    """
    Try to build a trend object with more than one axisfrom an HDF5 file.
    The loader returns a dict of SasData objects (here two, named 'sasentry01' and 'sasentry02')
    Need to think how to compare the Quantity objects
    """
    data = hdf_load_data(hdf_file) # dict of SasData objects
    trend = Trend(
        data=list(data.values()),
        trend_axes={'run_number': ['run'],
                    'title': ['title'],
                    'SDD1': ['sasinstrument', 'sasdetector01', 'SDD'],
                    'SDD2': ['sasinstrument', 'sasdetector02', 'SDD'],
                    'transmission': ['sastransmission_spectrum01', 'T'],
                    'wavelength': ['sastransmission_spectrum01', 'lambda']}
    )
    assert (not trend.is_manual_axis('run_number'))
    assert (trend.get_trend_values('run_number') == ['33837', '33837'])
    assert (trend.get_trend_values('title') == ['MH4_5deg_16T_SLOW', 'MH4_5deg_16T_SLOW'])
    assert (np.allclose(trend.get_trend_values('transmission')[0],
                        trend.get_trend_values('transmission')[1]))
    assert (np.allclose(trend.get_trend_values('wavelength')[0],
          trend.get_trend_values('wavelength')[1]))

@pytest.mark.parametrize('directory_name', [mumag_test_directories[1]])
def test_trend_build_from_ascii_with_manual_axis(directory_name: str):
    """
    Try to build a trend object from an ASCII file including a manual axis.
    """
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
        trend_axes={'index': ['magnetic', 'counting_index'],
                    'field': ['magnetic', 'applied_magnetic_field'],
                    'manual_temp': np.linspace(300, 350, len(data)).tolist()}
    )
    assert (not trend.is_manual_axis('index'))
    assert (trend.is_manual_axis('manual_temp'))

@pytest.mark.parametrize('directory_name', mumag_test_directories)
def test_trend_build_interpolate(directory_name: str):
    """
    Try to build a trend object on the MuMag datasets.
    and interpolates the data to match the Q axes.
    Maybe confusing to have here data axes ('Q', 'I', 'dI') 
    and trend axes ('field')?
    """
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
        trend_axes={'field': ['magnetic', 'applied_magnetic_field']}
    )
    # Initially, the q axes in this data don't exactly match
    to_interpolate_on = 'Q'
    assert not trend.all_axis_match(to_interpolate_on)
    interpolated_trend = trend.interpolate(to_interpolate_on)
    assert interpolated_trend.all_axis_match(to_interpolate_on)

def test_trend_q_axis_match():
    """
    Try to build a trend object on the custom test dataset
    and check if the Q axes match.
    But the file contents are skipped, so 'Q' and 'I' are zeroes!
    """
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
        trend_axes={'index': ['magnetic', 'counting_index']}
    )
    assert trend.all_axis_match('Q')
