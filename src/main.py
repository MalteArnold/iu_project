"""
Project for the Modul "Programmieren mit Python"
Author: Malte Arnold
"""

import os
import logging
import pandas as pd
from centrallogger import CentralLogger
from database import Database
from visualizer import Visualizer
from calculator import Calculator

# Path to this script
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

# Real path to the data directory
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
IDEAL_PATH = os.path.join(DATA_DIR, "ideal.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")


def main():
    """
    Main function to execute the data comparison and validation script
    """

    # Create a central logging object
    logger = CentralLogger(logging.INFO)
    logger.logger.info("###################################")
    logger.logger.info("### Starting Script ###")

    # Load .csv files into dataframes
    train_data = pd.read_csv(TRAIN_PATH)
    train_data.name = "train.csv"
    logger.logger.info("Creating dataframe: train")

    ideal_data = pd.read_csv(IDEAL_PATH)
    ideal_data.name = "ideal.csv"
    logger.logger.info("Creating dataframe: ideal")

    # Load the test data into a dataframe and sort it ascending by the column "x"
    test_data = pd.read_csv(TEST_PATH)
    test_data.name = "test.csv"
    sorted_test_data = test_data.sort_values(by="x", ignore_index=True)
    sorted_test_data.name = "sorted_test.csv"
    logger.logger.info("Creating sorted dataframe: test")

    # Create a database and insert the dataframes as tables
    database = Database()
    database.create_connection("iu-project")
    database.table_from_dataframe(train_data, train_data.name)
    database.table_from_dataframe(ideal_data, ideal_data.name)
    database.table_from_dataframe(test_data, test_data.name)
    database.table_from_dataframe(sorted_test_data, sorted_test_data.name)

    # Create visualization of train_data
    train_visualizer = Visualizer(train_data)
    train_visualizer.plot_data()
    train_visualizer.show_plot()

    # Calculate the best fits from the train_data and the ideal_data
    calculator = Calculator(train_data, ideal_data)
    result = calculator.least_squares()
    logger.logger.info("Calculating best fits")
    logger.logger.info(f"Result: {result}")

    # Create a table for the calculation results
    result_best_fits = pd.DataFrame(result)
    result_best_fits.name = "result best fits"
    database.table_from_dataframe(result_best_fits, result_best_fits.name)

    # Create a table for the best fits
    best_fits = database.table_bestfits(ideal_data, result_best_fits)
    best_fits.name = "best fits"
    database.table_from_dataframe(best_fits, best_fits.name)

    # Create a visualization of the best fits
    best_fits_visualizer = Visualizer(best_fits)
    best_fits_visualizer.plot_data()
    best_fits_visualizer.show_plot()

    # Create a visualization of the best fits and the train data in one plot
    best_fits_train_visualizer = Visualizer(best_fits, train_data)
    best_fits_train_visualizer.plot_selection()
    best_fits_train_visualizer.show_plot()

    # Validate the best fits with the test_data
    validation_result = calculator.validation(train_data, best_fits, sorted_test_data)
    logger.logger.info("validation_result: " + str(validation_result))

    logger.logger.info("### Finished script ###")
    logger.logger.info("###################################")


if __name__ == "__main__":
    main()
