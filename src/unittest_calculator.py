"""
Project for the Modul "Programmieren mit Python"
Author: Malte Arnold
"""

import unittest
import os
import pandas as pd
from calculator import Calculator

# Path to this script
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

# Real path to the data directory
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
IDEAL_PATH = os.path.join(DATA_DIR, "ideal.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")


class UnitTestCalculator(unittest.TestCase):
    """
    A class to perform unit tests on the Calculator class
    """

    def test_least_squares_with_empty_arguments(self):
        """
        Test least squares method with empty arguments
        """

        # Instantiate the object to be tested
        calc = Calculator(pd.DataFrame(), pd.DataFrame())

        # Test the least_squares method with empty arguments
        result = calc.least_squares()
        expected_result = []

        self.assertEqual(expected_result, result)

    def test_least_squares_with_arguments(self):
        """
        Test least squares method with real data arguments
        """

        # Instantiate the object to be tested with real data
        train_data = pd.read_csv(TRAIN_PATH)
        ideal_data = pd.read_csv(IDEAL_PATH)
        calc = Calculator(train_data, ideal_data)

        # Test the least_squares method with real data
        result = calc.least_squares()
        expected_result = [
            {
                "train_data_y": 1,
                "ideal_data_y": 36,
                "minimal_deviation_value": 33.71178854422825,
                "minimal_deviation_index": 35,
            },
            {
                "train_data_y": 2,
                "ideal_data_y": 11,
                "minimal_deviation_value": 32.62893138832119,
                "minimal_deviation_index": 10,
            },
            {
                "train_data_y": 3,
                "ideal_data_y": 2,
                "minimal_deviation_value": 33.11847188090259,
                "minimal_deviation_index": 1,
            },
            {
                "train_data_y": 4,
                "ideal_data_y": 33,
                "minimal_deviation_value": 31.752431101394777,
                "minimal_deviation_index": 32,
            },
        ]

        self.assertEqual(expected_result, result)

    def test_max_deviation_best_fits_to_test_data_with_no_arguments(self):
        """
        Test max deviation from best fits to test data with empty arguments
        """

        # Instantiate the object to be tested
        calc = Calculator(pd.DataFrame(), pd.DataFrame())

        # Test the max_deviation_best_fits_to_test_data method with empty arguments
        result = calc.max_deviation_best_fits_to_test_data(
            pd.DataFrame(), pd.DataFrame()
        )
        expected_result = []

        self.assertEqual(expected_result, result)

    def test_max_deviation_train_data_to_best_fits_with_no_arguments(self):
        """
        Test max deviation from train data to best fits with empty arguments
        """

        # Instantiate the object to be tested
        calc = Calculator(pd.DataFrame(), pd.DataFrame())

        # Test the max_deviation_train_data_to_best_fits method with empty arguments
        result = calc.max_deviation_train_data_to_best_fits(
            pd.DataFrame(), pd.DataFrame()
        )
        expected_result = []

        self.assertEqual(expected_result, result)


if __name__ == "__main__":
    unittest.main()
