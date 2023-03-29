"""
    Unit tests for loading data files using the extension registry
"""

import logging
import unittest
import os
import shutil
import numpy as np

from sasdata.dataloader.loader import Registry as Loader

logger = logging.getLogger(__name__)


def find(filename):
    return os.path.join(os.path.dirname(__file__), 'data', filename)


class ExtensionRegistryTests(unittest.TestCase):

    def setUp(self):
        self.valid_file = find("valid_cansas_xml.xml")
        self.valid_file_url = "https://github.com/SasView/sasdata/raw/master/test/sasdataloader/data/valid_cansas_xml.xml"
        self.valid_file_wrong_known_ext = find("valid_cansas_xml.txt")
        self.valid_file_wrong_unknown_ext = find("valid_cansas_xml.xyz")
        shutil.copyfile(self.valid_file, self.valid_file_wrong_known_ext)
        shutil.copyfile(self.valid_file, self.valid_file_wrong_unknown_ext)
        self.invalid_file = find("cansas1d_notitle.xml")

        self.valid_hdf_file = find("MAR07232_rest.h5")
        self.valid_hdf_url = "https://github.com/SasView/sasdata/raw/master/test/sasdataloader/data/MAR07232_rest.h5"

        self.loader = Loader()

    def test_wrong_known_ext(self):
        """
        Load a valid CanSAS XML file that has the extension '.txt', which is in
        the extension registry. Compare the results to loading the same file
        with the extension '.xml'
        """
        correct = self.loader.load(self.valid_file)
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
        correct = self.loader.load(self.valid_file)
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
        remote = self.loader.load(self.valid_hdf_url)
        local = self.loader.load(self.valid_hdf_file)
        # Ensure the string representation of the file contents match
        self.assertEqual(str(local[0]), str(remote[0]))

    def tearDown(self):
        if os.path.isfile(self.valid_file_wrong_known_ext):
            os.remove(self.valid_file_wrong_known_ext)
        if os.path.isfile(self.valid_file_wrong_unknown_ext):
            os.remove(self.valid_file_wrong_unknown_ext)
