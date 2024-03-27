"""
Project for the Modul "Programmieren mit Python"
Author: Malte Arnold
"""

import logging
import pandas as pd
import numpy as np
import math
from centrallogger import CentralLogger

logger = CentralLogger(logging.INFO)


class Calculator:
    """
    A class to represent all calculations for the project
    """

    def __init__(self, *args):
        """
        Constructor for the Calculator class
                Parameters:
                    args[0] (dataframe): source data for calculations
                    args[1] (dataframe): data to compare against
        """

        logger.logger.info("Creating calculator object")
        self.source_data = args[0]
        self.selected_data = args[1]

    def least_squares(self):
        """
        Calculates the minimal deviation of 2 dataframes

        Returns:
            list: a list of dictionaries with results for each calculation
                Each dictionary contains:
                    - 'train_data_y': the column number in the source data
                    - 'ideal_data_y': the column number in the selected data with minimal deviation
                    - 'minimal_deviation_value': the minimal deviation value
                    - 'minimal_deviation_index': the index corresponding to the minimal deviation
        """

        logger.logger.info("Calculating least squares")

        result_dict_list = []

        # Iterate over each column in the source data (excluding the 'x' column)
        for column_train in self.source_data.columns[1:]:
            # Extract the y-values as numpy arrays
            train_y_array = self.source_data[column_train].to_numpy()

            # Calculate the sum of squared deviations for each y-value in the selected data
            sum_array = np.sum(
                (train_y_array[:, np.newaxis] - self.selected_data.iloc[:, 1:]) ** 2,
                axis=0,
            )

            # Find the index and value of the minimum deviation
            minimal_deviation_value = np.min(sum_array)
            minimal_deviation_index = np.argmin(sum_array)
            ideal_data_y = minimal_deviation_index + 1

            # Store the results in a dictionary
            result_dict = {
                "train_data_y": int(
                    column_train[1:]
                ),  # Extract the number from the column label
                "ideal_data_y": ideal_data_y,
                "minimal_deviation_value": minimal_deviation_value,
                "minimal_deviation_index": minimal_deviation_index,
            }

            # Log the results for each iteration
            logger.logger.info(
                f"### Train-function: y{result_dict['train_data_y']} ###"
            )
            logger.logger.info(
                f"Result for train_data_y = {result_dict['train_data_y']}"
            )
            logger.logger.info(f"Ideal data y: {result_dict['ideal_data_y']}")
            logger.logger.info(
                f"Minimal deviation value: {result_dict['minimal_deviation_value']}"
            )
            logger.logger.info(
                f"Minimal deviation index: {result_dict['minimal_deviation_index']}"
            )

            # Append the result dictionary to the list
            result_dict_list.append(result_dict)

        return result_dict_list

    # calculation of M
    def max_deviation_best_fits_to_test_data(self, best_fits_data, test_data):
        """
        Calculates the maximal deviation of best_fits to test_data
                Parameters:
                        best_fits_data (dataframe): a pandas dataframe
                        test_data (dataframe): a pandas dataframe
                Returns:
                        result_dict_list (list): a list of dictionaries with the results of calculation
        """

        result_dict_list = []

        # check if the dataframes are empty
        if best_fits_data.empty | test_data.empty:
            return result_dict_list

        logger.logger.info(
            "Get column values from best_fits_data and corresponding test_data"
        )

        for column_best in best_fits_data:
            if column_best == "x":
                continue

            result_array = 0.0
            result_dict = {
                "best_fits_data": column_best,
                "M": column_best,
                "maximal_deviation_value": 0.0,
                "maximal_deviation_index": 0,
            }

            for row in test_data.index:
                logger.logger.debug("Column best: " + str(column_best))
                logger.logger.debug("test_data_row_index: " + str(row))

                # .loc[row, column] = get value of location
                x_value_test_data = test_data.loc[row, "x"]
                y_value_test_data = test_data.loc[row, "y"]
                logger.logger.debug("x_value: " + str(x_value_test_data))
                logger.logger.debug("y_value: " + str(y_value_test_data))
                logger.logger.debug(
                    "y_value to find from x = " + str(x_value_test_data)
                )

                # get index of x_value in the best_fits_Data
                selection = best_fits_data["x"]
                position_index = pd.Index(selection).get_loc(x_value_test_data)

                # .loc[row, column] = get value of location
                y_value_best_fits = best_fits_data.loc[position_index, column_best]
                if y_value_best_fits is None:
                    logger.info(
                        "no y value found in best_fits for x: " + str(x_value_test_data)
                    )
                    continue

                logger.logger.debug(
                    "column : "
                    + str(column_best)
                    + " --> position_in_column: "
                    + str(position_index)
                    + " --> y_value: "
                    + str(y_value_best_fits)
                )

                logger.logger.debug(
                    "x value is "
                    + str(x_value_test_data)
                    + ", calculation of deviation ..."
                )
                result = np.subtract(y_value_best_fits, y_value_test_data)
                logger.logger.debug(
                    "Result: "
                    + str(y_value_best_fits)
                    + " - "
                    + str("(")
                    + str(y_value_test_data)
                    + str(")")
                    + " = "
                    + str(result)
                )
                result_array = np.append(result_array, result)
                logger.logger.debug("#########################################")

            index_to_delete = 0
            new_result_array = np.delete(result_array, index_to_delete)
            maximal_deviation = np.max(new_result_array)
            result_dict["maximal_deviation_value"] = maximal_deviation
            maximal_deviation_index = np.argmax(new_result_array)
            result_dict["maximal_deviation_index"] = maximal_deviation_index

            result_dict_list.append(result_dict)
        logger.logger.info("result_dict: " + str(result_dict_list))
        return result_dict_list

    def max_deviation_train_data_to_best_fits(self, train_data, best_fits_data):
        """
        Calculates the maximal deviation of train_data to best_fits_data
                Parameters:
                        train_data (dataframe): a pandas dataframe
                        best_fits_data (dataframe): a pandas dataframe
                Returns:
                        result_dict_list (list): a list of dictionaries with results of calculation
        """

        result_dict_list = []

        # check whether the dataframe are empty
        if train_data.empty | best_fits_data.empty:
            return result_dict_list

        columns_train_data = len(train_data.columns)
        rows_train_data = len(train_data.index)

        logger.logger.info(
            "Get column values from train_data and corresponding best_fits_data"
        )

        for column_train in range(1, columns_train_data, 1):
            result_array = 0.0
            result_dict = {
                "train_data": 0,
                "best_fits_data": 0,
                "N": 0,
                "maximal_deviation_value": 0.0,
                "maximal_deviation_index": 0,
            }

            # create array from y column values
            current_train_column = train_data.columns[column_train]
            train_y_array = np.array(train_data[current_train_column])
            result_dict["train_data"] = current_train_column
            current_best_fits_column = best_fits_data.columns[column_train]
            best_fits_y_array = np.array(best_fits_data[current_best_fits_column])
            result_dict["best_fits_data"] = current_best_fits_column
            result_dict["N"] = current_best_fits_column

            # 400 rows to subtract
            for row in range(0, rows_train_data, 1):
                result = np.subtract(train_y_array[row], best_fits_y_array[row])
                result_array = np.append(result_array, result)

            index_to_delete = 0
            new_result_array = np.delete(result_array, index_to_delete)
            maximal_deviation = np.max(new_result_array)
            result_dict["maximal_deviation_value"] = maximal_deviation
            maximal_deviation_index = np.argmax(new_result_array)
            result_dict["maximal_deviation_index"] = maximal_deviation_index

            result_dict_list.append(result_dict)
        logger.logger.info("result_dict: " + str(result_dict_list))
        return result_dict_list

    def validation(self, train_data, best_fits_data, test_data):
        """
        Calculates the validation condition, M < (sqrt(2)) * N
                Parameters:
                        train_data (dataframe): a pandas dataframe
                        best_fits_data (dataframe): a pandas dataframe
                        test_data (dataframe): a pandas dataframe
                Returns:
                        result_dict_list (list): a list of dictionaries with results of calculation
        """

        result_dict_list = []
        logger.logger.info("Validation calculation")

        result_m_list = self.max_deviation_best_fits_to_test_data(
            best_fits_data, test_data
        )
        result_n_list = self.max_deviation_train_data_to_best_fits(
            train_data, best_fits_data
        )

        logger.logger.info("Calculation of M < (sqrt(2)) * N")

        for item in range(0, result_m_list.__len__(), 1):
            m_max_deviation = result_m_list[item].get("maximal_deviation_value")
            logger.logger.info("max M deviation: " + str(m_max_deviation))
            n_max_deviation = result_n_list[item].get("maximal_deviation_value")
            logger.logger.info("max N deviation: " + str(n_max_deviation))
            n_condition = n_max_deviation * math.sqrt(2)
            logger.logger.info("sqrt(2)*N = " + str(n_condition))

            if m_max_deviation < n_condition:
                logger.logger.info(
                    "M: "
                    + str(m_max_deviation)
                    + " is smaller than sqrt(2)*N: "
                    + str(n_condition)
                )

        return result_dict_list
