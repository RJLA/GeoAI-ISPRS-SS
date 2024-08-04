"""
This module provides a class for performing various operations on pandas
DataFrames, including converting raster arrays to DataFrames, removing duplicate
rows, and splitting data into training and test sets with stratified sampling.

Classes:
    DataFrameOperations: A class that provides operations for converting raster
    arrays to pandas DataFrames.
"""

__author__ = "Reginald Jay L. Argamosa"
__version__ = "0.1.0"
__email__ = "regi.argamosa@gmail.com"
__license__ = (
    "Reginald Jay L. Argamosa Personal Use License: See LICENSE file for details"
)

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataFrameOperations:
    """
    A class that provides operations for converting raster arrays to pandas
    DataFrames, removing duplicate rows, and splitting data into training and
    test sets with stratified sampling.

    Attributes:
        None

    Methods:
        convert_to_df: Convert a raster array to a pandas DataFrame.
        remove_duplicate_rows: Remove duplicate rows from a DataFrame.
        split_data: Split data into training and test sets with stratified sampling.
    """

    def convert_to_df(self, array: np.ndarray, column_name: str) -> pd.DataFrame:
        """
        Convert a raster array to a pandas DataFrame

        Args:
            array (np.np.ndarray): The input raster array to be converted.
            column_name (str): The name to be used for the DataFrame's column.

        Returns:
            pd.DataFrame: A DataFrame containing the raster array.
        """
        df = pd.DataFrame(array, columns=[column_name])
        return df

    def remove_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate rows from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame from which to remove duplicates.

        Returns:
            pd.DataFrame: A DataFrame with duplicate rows removed.
        """
        return df.drop_duplicates()

    def split_data(
        self,
        df: pd.DataFrame,
        label_column: str,
        test_size: float = 0.2,
        random_state: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data into training and test sets with stratified sampling.

        Args:
            df (pd.DataFrame): DataFrame containing the data. label_column
            label_column (str): The column containing the labels.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data
            before applying the split.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple
            containing the training and test sets for the features and labels.
        """

        X = df.drop(label_column, axis=1)
        y = df[label_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        return X_train, X_test, y_train, y_test
