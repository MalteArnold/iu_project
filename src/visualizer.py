"""
Project for the Modul "Programmieren mit Python"
Author: Malte Arnold
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from centrallogger import CentralLogger

logger = CentralLogger(logging.INFO)


class Visualizer:
    """
    A class to represent the data visualization
    """

    def __init__(self, *args):
        """
        Constructor for the Calculator class
                Parameters:
                    args[0] (dataframe): main data source for the plot
                    args[1] (dataframe, optional): selection for which the data is visualized
        """

        logger.logger.info("Creating visualizer object")
        self.source_data = args[0]

        if len(args) > 1 and isinstance(args[1], pd.DataFrame):
            self.selected_data = args[1]
        else:
            self.selected_data = pd.DataFrame()

    def plot_data(self):
        """
        Visualizes the data
        """

        logger.logger.info("Visualizing data")

        style.use("ggplot")
        plt.figure(figsize=(16, 9))

        for y_index in self.source_data:
            if y_index != "x":
                plt.plot(
                    self.source_data["x"],
                    self.source_data[y_index],
                    label=y_index,
                    linewidth=2,
                )

        plt.legend(loc=(1.01, 0))
        plt.grid(True, color="k", linestyle=":")
        plt.title(self.source_data.name)
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        logger.logger.info(f"Creating plot: {self.source_data.name}")
        return plt.gca()

    def plot_selection(self):
        """
        Creates a plot from the source dataframe and a selection from another dataframe
                Parameters:
                        style_name (str, optional): matplotlib style to be used
                        figsize (tuple, optional): figure size
                Returns:
                        the created plot
        """

        style.use("ggplot")
        plt.figure(figsize=(16, 9))

        logger.logger.info(f"Creating plot for source DataFrame")
        for columns in self.source_data:
            if columns != "x":
                plt.plot(
                    self.source_data["x"], self.source_data[columns], label=columns
                )

        logger.logger.info(f"Creating plot for selection")
        for columns in self.selected_data:
            if columns != "x":
                plt.plot(
                    self.selected_data["x"],
                    self.selected_data[columns],
                    label=columns,
                    linewidth=2,
                    alpha=0.8,
                )

        plt.legend(loc=(1.01, 0))
        plt.grid(True, color="k", linestyle=":")
        plt.title("Best fits")
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        logger.logger.info(f"Creating plot: Best fits")
        return plt.gca()

    def show_plot(self):
        """
        Displays the created plot
        """

        logger.logger.info("Showing plot")
        plt.show()
