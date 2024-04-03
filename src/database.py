"""
Project for the Modul "Programmieren mit Python"
Author: Malte Arnold
"""

import logging
import sqlalchemy as db
import pandas as pd
import numpy as np
import os
from centrallogger import CentralLogger

logger = CentralLogger(logging.INFO)


class Database:
    """
    A class to represent a database and methods to interact with the database
    """

    def __init__(self):
        """
        Constructor for the Calculator class
        """

        logger.logger.info("Creating database object")
        self.database_name = None
        self.connection_string = ""
        self.metadata = None
        self.engine = None

    def create_connection(self, database_name):
        """
        Creates a connection to a database
                Parameters:
                        database_name (str): name of the database to connect to
        """

        logger.logger.info(f"Creating connection to database: {database_name}")

        # Set database name and create connection string
        self.database_name = database_name
        password = os.environ["MYSQL_PASSWORD"]
        port = "3306"
        host = "localhost"
        self.connection_string = (
            f"mysql+pymysql://root:{password}@{host}:{port}/{database_name}"
        )
        logger.logger.info(f"Connection string: {self.connection_string}")

        # Create engine-object for database connection
        self.engine = db.create_engine(self.connection_string)
        logger.logger.info("Creating engine object")

        # Create connection to database
        try:
            self.connection = self.engine.connect()
            logger.logger.info("Creating connection object")
        except Exception as e:
            logger.logger.error(f"Error while creating connection to database: {e}")

        # Create metadata object
        self.metadata = db.MetaData()
        logger.logger.info("Creating metadata object")

    def table_from_dataframe(self, dataframe_object, table_name):
        """
        Creates a database table from a dataframe
                Parameters:
                        dataframe_object (dataframe): a pandas dataframe
                        table_name (str): name of the table
        """

        # Check if object is a pandas dataframe
        if not isinstance(dataframe_object, pd.DataFrame):
            logger.logger.error("Object is not a pandas dataframe")
            return

        # Create table from dataframe
        try:
            dataframe_object.to_sql(
                table_name,
                con=self.engine,
                schema="iu-project",
                index=False,
                if_exists="replace",
            )
            logger.logger.info(f"Creating database table: {table_name}")
        except Exception as e:
            logger.logger.error(f"Error while creating table from dataframe: {e}")

    def table_bestfits(self, ideal_data, best_fits_result):
        """
        Creates the 'best_fits' table from 'ideal_data' and 'best_fits_result'
                Parameters:
                        ideal_data (dataframe): a pandas dataframe
                        best_fits_result (dataframe): a pandas dataframe
                Returns:
                    result_data (dataframe): a pandasa dataframe with results of calculation
        """

        result_data = pd.DataFrame()

        # Create 400 random x values from -20.0 to 19.9, round to 0.1
        result_data["x"] = np.round(
            np.linspace(-20, 20, 400, endpoint=False), decimals=1
        )

        # Get the column values from ideal_data_y in best_fits_result
        best_fits_functions = best_fits_result["ideal_data_y"].values
        logger.logger.info(f"Best fitting functions: {best_fits_functions}")

        # Create a column for each best fitting function
        for function in best_fits_functions:
            result_column = "y" + str(function)
            data = ideal_data[result_column]
            result_data[result_column] = data.values
        return result_data

