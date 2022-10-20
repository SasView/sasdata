"""
    Unit tests for the unit conversion tool
"""
import unittest

from sasdata.data_util.nxsunit import Converter, standardize_units


class NXSUnitTests(unittest.TestCase):

    def setUp(self):
        self.converter = Converter(None)
        self.k_conv = Converter('nanoKelvins')
        self.sesans = Converter('A-2 cm-1')
        self.base_value = 123

    def test_initialization(self):
        self.assertEqual(self.converter.units, 'a.u.')
        self.assertEqual(self.converter.scalebase, 1)
        self.assertTrue(isinstance(self.converter.scalemap, list))
        self.assertEqual(len(self.converter.scalemap), 1)
        self.assertIn('None', self.converter.scalemap[0].keys())
        self.assertEqual(self.k_conv.units, 'nanoK')
        self.assertEqual(self.k_conv.scalebase, 1e-9)
        self.assertEqual(self.k_conv.scaleoffset, 0.0)
        self.assertEqual(len(self.k_conv.scalemap[0].keys()), 456)

    def testBasicUnits(self):
        # 10 nm^-1 = 1 inv Angstroms
        self.assertAlmostEqual(1, Converter('n_m^-1')(10, 'invA'))
        # Test different 1/A representations
        self.assertAlmostEqual(1, Converter('/A')(1, 'invA'))
        # 1.65 1/A = 1.65e10 1/m
        self.assertAlmostEqual(1.65e10, Converter('/A')(1.65, '/m'), 2)
        # 2000 mm = 2 m
        self.assertAlmostEqual(2.0, Converter('mm')(2000, 'm'))
        # 2.011 1/A = 2.011e10 1/m
        self.assertAlmostEqual(2.011e10, Converter('1/A')(2.011, "1/m"))
        # 3 us = 0.003 ms
        self.assertAlmostEqual(0.003, Converter('microseconds')(3, units='ms'))
        # 45 nK = 45 nK
        self.assertAlmostEqual(45, self.k_conv(45))
        # 1 K = 1e9 nK
        self.assertAlmostEqual(1, self.k_conv(1e9, 'K'))
        # 1800 s = 0.5 hr
        self.assertAlmostEqual(0.5, Converter('seconds')(1800, units='hours'))
        # 100 nm^-2 = 1 A^-2
        self.assertAlmostEqual(100, Converter('A^-2')(1.0, 'nm^-2'))

    def test_known_unknown_units(self):
        self.assertEqual(self.base_value,
                         self.converter(self.base_value, units='a.u.'))
        self.assertEqual(self.base_value,
                         Converter('arbitrary')(self.base_value, units='a.u.'))
        self.assertEqual(self.base_value,
                         Converter('cts')(self.base_value, units='a.u.'))
        self.assertEqual(self.base_value,
                         Converter('Counts')(self.base_value, units='a.u.'))
        self.assertEqual(self.base_value,
                         Converter('Unknown')(self.base_value, units='a.u.'))
        self.assertEqual(self.base_value,
                         Converter('a.u.')(self.base_value, units='a.u.'))
        self.assertEqual(self.base_value,
                         Converter('A.U.')(self.base_value, units='a.u.'))
        self.assertEqual(self.base_value,
                         Converter('arbitrary units')(self.base_value, units='a.u.'))
        self.assertEqual(1, self.converter.scalebase)

    def test_se_units(self):

        # Test many variations of the unit structure
        self.assertAlmostEqual(1, self.sesans(1, 'Å^{-2} cm^{-1}'))
        # Check units can be scaled independently
        self.assertAlmostEqual(100, self.sesans(1, 'A-2 m-1'))
        self.assertAlmostEqual(100, self.sesans(1, 'nm-2 cm-1'))
        # Check scaling works when modifying multiple units
        self.assertAlmostEqual(10000, self.sesans(1, 'nm-2 m-1'))
        # Check forward and backward scaling matches
        self.assertAlmostEqual(1.0, self.sesans(Converter('nm-2 m-1')(1.0, 'Å^{-2} cm^{-1}'), 'nm-2 m-1'))

    def test_temp_units(self):
        # Test scaling with offset
        self.assertAlmostEqual(-273.15, Converter('K')(0, '℃'))
        self.assertAlmostEqual(273.15, Converter('℃')(0, 'K'))
        self.assertAlmostEqual(0, Converter('F')(-459.67, 'R'))
        self.assertAlmostEqual(459.67, Converter('F')(0, 'R'))
        self.assertAlmostEqual(0, Converter('K')(0, 'R'))
        # Test forward and backward conversions
        self.assertAlmostEqual(0, Converter('℃')(Converter('K')(0, '℃'), 'K'))
        self.assertAlmostEqual(0, Converter('R')(Converter('F')(0, 'R'), 'F'))

    def test_incompatible_units(self):
        # Ensure incompatible units cannot be scaled
        self.assertRaises(ValueError, self.k_conv, value=1.0, units='m')
        self.assertRaises(ValueError, self.converter, value=1.0, units='m')

    def test_unit_structures(self):
        # Both should return None
        self.assertEqual(standardize_units(None), standardize_units('None'))
        # Test substitutions
        self.assertEqual(['nK'], standardize_units('nKelvin'))
        # Capitalization standardization
        self.assertEqual(standardize_units('seconds'),
                         standardize_units('SECONDS'))
        # Underscore standardization
        self.assertEqual(standardize_units('n_m'),
                         standardize_units('nm'))
        # Different '*' standardizations
        self.assertEqual(standardize_units('pico*meter'),
                         standardize_units('picometer'))
        self.assertEqual(standardize_units('Å^2 Kelvin^4'),
                         standardize_units('Ang^{2}*K^{4}'))
        # US vs. European spellings
        self.assertEqual(standardize_units('meters'),
                         standardize_units('metres'))
        # Multiple units
        self.assertEqual(['nanoK^{-4}', 'cm^{-1}', 'Å^{-2}'],
                         standardize_units('nanoKelvin-4 invcm/angstrom^2'))
        # Numerator vs. Denominator
        self.assertEqual(standardize_units('A^2/nanoKelvin^4'),
                         ['A^{2}', 'nanoK^{-4}'])
        # Tackle parentheses
        self.assertEqual(standardize_units('(A^2 B^2)/(C^2)'),
                         ['A^{2}', 'B^{2}', 'C^{-2}'])
        # Multiple divisions
        self.assertEqual(standardize_units('A/B/C'), ['A', 'B^{-1}', 'C^{-1}'])
