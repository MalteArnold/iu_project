"""
Project for the Modul "Programmieren mit Python"
Author: Malte Arnold
"""

import logging
import os

# Path to this script
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

# Real path to the logging directory
LOGGING_DIR = os.path.join(SCRIPT_DIR, "logging")


class CentralLogger:
    """
    A class to represent a central logger for the project
    """

    def __init__(self, loglevel, logdir=LOGGING_DIR):
        """
        Constructor for the CentralLogger class
        """

        self.loglevel = loglevel
        self.logdir = logdir
        self.logname = "central.log"
        self.logfile = os.path.join(self.logdir, self.logname)

        # Create logger object if no handlers are present
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.setLevel(self.loglevel)

            # Create formatter object for the log messages
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            # Create filehandler for the log messages
            filehandler = logging.FileHandler(self.logfile, mode="a")
            filehandler.setLevel(self.loglevel)
            filehandler.setFormatter(formatter)

            # Create streamhandler for the log messages
            streamhandler = logging.StreamHandler()
            streamhandler.setLevel(self.loglevel)
            streamhandler.setFormatter(formatter)

            # Add handlers to the logger
            self.logger.addHandler(filehandler)
            self.logger.addHandler(streamhandler)

    def error(self, message, *args, **kwargs):
        """
        Log an error message
        """

        self.logger.error(message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        """
        Log an info message
        """

        self.logger.info(message, *args, **kwargs)


# Example usage:
# logger = CentralLogger(logging.INFO, "../logging")
# logger.logger.info("Logger test")
