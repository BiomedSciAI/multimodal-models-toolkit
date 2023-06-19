"""Unit test for complex_module.core."""

__copyright__ = """
LICENSED INTERNAL CODE. PROPERTY OF IBM.
IBM Research Licensed Internal Code
(C) Copyright IBM Corp. 2021
ALL RIGHTS RESERVED
"""

import unittest
from mmmt.data.graph.general_file_loader import GeneralFileLoader


class GeneralFileLoaderTestCase(unittest.TestCase):
    """GeneralFileLoaderTestCase class."""

    def setUp(self):
        """Setting up the test."""
        pass

    def test_incorrect_splits(self):
        """Test that supported datasets return the correct types."""
        with self.assertRaises(AssertionError):
            loader = GeneralFileLoader(None, [0.6, 0.3, 0.4], 0)

    def test_not_enough_parameters(self):
        """Test that a non supported dataset raises an error"""
        with self.assertRaises(TypeError):
            GeneralFileLoader()

    def tearDown(self):
        """Tear down the tests."""
        pass
