"""
    Unit tests for loading data files using the extension registry
"""

import logging
import unittest
import os
import shutil
import numpy as np

from sasdata.dataloader.loader import Registry as Loader
from sasdata.dataloader.loader import Loader as LoaderMain

logger = logging.getLogger(__name__)

BASE_URL = 'https://github.com/SasView/sasdata/raw/master/test/sasdataloader/data/'


def find(filename):
    return os.path.join(os.path.dirname(__file__), 'data', filename)


class ExtensionRegistryTests(unittest.TestCase):

    def setUp(self):
        # Local and remote files to compare loading
        # NXcanSAS
        self.valid_hdf_file = find("MAR07232_rest.h5")
        self.valid_hdf_url = BASE_URL + "MAR07232_rest.h5"
        # canSAS XML
        self.valid_xml_file = find("valid_cansas_xml.xml")
        self.valid_xml_url = BASE_URL + "valid_cansas_xml.xml"
        # ASCII Text
        self.valid_txt_file = find("avg_testdata.txt")
        self.valid_txt_url = BASE_URL + "avg_testdata.txt"
        # ABS Text
        self.valid_abs_file = find("ascii_test_4.abs")
        self.valid_abs_url = BASE_URL + "ascii_test_4.abs"
        # DAT 2D NIST format
        self.valid_dat_file = find("detector_square.dat")
        self.valid_dat_url = BASE_URL + "detector_square.dat"
        # Anton Parr SAXSess PDH format
        self.valid_pdh_file = find("Anton-Paar.pdh")
        self.valid_pdh_url = BASE_URL + "Anton-Paar.pdh"

        self.valid_file_wrong_known_ext = find("valid_cansas_xml.txt")
        self.valid_file_wrong_unknown_ext = find("valid_cansas_xml.xyz")
        shutil.copyfile(self.valid_xml_file, self.valid_file_wrong_known_ext)
        shutil.copyfile(self.valid_xml_file, self.valid_file_wrong_unknown_ext)
        self.invalid_file = find("cansas1d_notitle.xml")

        self.loader = Loader()

    def test_wrong_known_ext(self):
        """
        Load a valid CanSAS XML file that has the extension '.txt', which is in
        the extension registry. Compare the results to loading the same file
        with the extension '.xml'
        """
        correct = self.loader.load(self.valid_xml_file)
        wrong_ext = self.loader.load(self.valid_file_wrong_known_ext)
        self.assertEqual(len(correct), 1)
        self.assertEqual(len(wrong_ext), 1)
        correct = correct[0]
        wrong_ext = wrong_ext[0]

        self.assertTrue(np.all(correct.x == wrong_ext.x))
        self.assertTrue(np.all(correct.y == wrong_ext.y))
        self.assertTrue(np.all(correct.dy == wrong_ext.dy))

    def test_wrong_unknown_ext(self):
        """
        Load a valid CanSAS XML file that has the extension '.xyz', which isn't
        in the extension registry. Compare the results to loading the same file
        with the extension '.xml'
        """
        correct = self.loader.load(self.valid_xml_file)
        wrong_ext = self.loader.load(self.valid_file_wrong_unknown_ext)
        self.assertEqual(len(correct), 1)
        self.assertEqual(len(wrong_ext), 1)
        correct = correct[0]
        wrong_ext = wrong_ext[0]

        self.assertTrue(np.all(correct.x == wrong_ext.x))
        self.assertTrue(np.all(correct.y == wrong_ext.y))
        self.assertTrue(np.all(correct.dy == wrong_ext.dy))

    def test_data_reader_exception(self):
        """
        Load a CanSAS XML file that doesn't meet the schema, and check errors
        are set correctly
        """
        data = self.loader.load(self.invalid_file)
        self.assertEqual(len(data), 1)
        data = data[0]
        self.assertEqual(len(data.errors), 1)

        err_msg = data.errors[0]
        self.assertTrue("does not fully meet the CanSAS v1.x specification" in err_msg)

    def test_compare_remote_file_to_local(self):
        """Load the same file from a local directory and a remote URL and compare data objects."""
        # ASCII Text file loading
        remote_txt = self.loader.load(self.valid_txt_url)
        local_txt = self.loader.load(self.valid_txt_file)
        # Ensure the string representation of the file contents match
        self.assertEqual(str(local_txt[0]), str(remote_txt[0]))
        # NXcanSAS file loading
        local_hdf = self.loader.load(self.valid_hdf_file)
        remote_hdf = self.loader.load(self.valid_hdf_url)
        # Ensure the string representation of the file contents match
        self.assertEqual(str(local_hdf[0]), str(remote_hdf[0]))
        # canSAS XML file loading
        local_xml = self.loader.load(self.valid_xml_file)
        remote_xml = self.loader.load(self.valid_xml_url)
        # Ensure the string representation of the file contents match
        self.assertEqual(str(local_xml[0]), str(remote_xml[0]))
        # ABS file loading
        local_abs = self.loader.load(self.valid_abs_file)
        remote_abs = self.loader.load(self.valid_abs_url)
        # Ensure the string representation of the file contents match
        self.assertEqual(str(local_abs[0]), str(remote_abs[0]))
        # DAT file loading
        local_dat = self.loader.load(self.valid_dat_file)
        remote_dat = self.loader.load(self.valid_dat_url)
        # Ensure the string representation of the file contents match
        self.assertEqual(str(local_dat[0]), str(remote_dat[0]))
        # PDH file loading
        local_pdh = self.loader.load(self.valid_pdh_file)
        remote_pdh = self.loader.load(self.valid_pdh_url)
        # Ensure the string representation of the file contents match
        self.assertEqual(str(local_pdh[0]), str(remote_pdh[0]))

    def test_load_simultaneously(self):
        """Load a list of files, not just a single file, and ensure the content matches"""
        loader = LoaderMain()
        local_txt = loader.load(self.valid_txt_file)
        local_hdf = loader.load(self.valid_hdf_file)
        local_xml = loader.load(self.valid_xml_file)
        strings = [str(local_txt[0]), str(local_hdf[0]), str(local_xml[0])]
        all_files = loader.load([self.valid_xml_file, self.valid_hdf_file, self.valid_txt_file])
        for file in all_files:
            self.assertTrue(str(file) in strings)

    def tearDown(self):
        if os.path.isfile(self.valid_file_wrong_known_ext):
            os.remove(self.valid_file_wrong_known_ext)
        if os.path.isfile(self.valid_file_wrong_unknown_ext):
            os.remove(self.valid_file_wrong_unknown_ext)
