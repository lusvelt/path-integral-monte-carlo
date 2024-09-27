"""
This module contains functions for saving and loading data.
"""

import pickle

FILES_DIR = "../data/"


def save(data: object, filename: str):
    """
    Saves data to a file

    Args:
        data (object): The python object to be saved
        filename (str): The name of the file to be created, without extension
    """
    with open(FILES_DIR + filename + ".pickle", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()


def load(filename: str) -> object:
    """
    Loads data stored into a file

    Args:
        filename (str): The name of the file where the data is stored

    Returns:
        object: The python object containing the retrieved data
    """
    with open(FILES_DIR + filename + ".pickle", "rb") as f:
        data = pickle.load(f)
        f.close()
        return data
