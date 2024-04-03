"""
Project for the Modul "Programmieren mit Python"
Author: Malte Arnold
"""

import unittest
from unittest.mock import patch
from database import Database


class UnitTestDatabase(unittest.TestCase):
    """
    A class to perform unit tests for the Database class
    """

    def setUp(self):
        """
        Set up method to instantiate a Database object before each test method
        """

        self.database = Database()

    @patch("os.environ", {"MY_SQL_PASSWORD": "PASSWORT"})
    def test_create_connection(self):
        """
        Unit test method to test the create_connection method of the Database class
        This method ensures that the create_connection method successfully establishes a connection to the MySQL database
        """

        self.database.create_connection("iu-project")
        self.assertEqual(self.database.database_name, "iu-project")
        self.assertIsNotNone(self.database.engine)
        self.assertIsNotNone(self.database.metadata)


if __name__ == "__main__":
    unittest.main()
