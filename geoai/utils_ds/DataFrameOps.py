from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataFrameOperations:
    """
    A class that provides operations for converting
    raster arrays to pandas DataFrames.
    """

    def convert_to_df(self, array: np.ndarray, column_name: str) -> pd.DataFrame:
        """
        Convert a raster array to a pandas DataFrame

        Args:
            array (np.np.ndarray): The input raster array to be converted.
            column_name (str): The name to be used for the DataFrame's column.

        Returns:
            df (DataFrame): The DataFrame containing the raster data.
        """
        print(f"Converting raster array to DataFrame with column name {column_name}")
        df = pd.DataFrame(array, columns=[column_name])
        return df

    def remove_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate rows from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame from which to remove
            duplicates.

        Returns:
            pd.DataFrame: A DataFrame with duplicate rows removed.
        """
        return df.drop_duplicates()

    from sklearn.model_selection import train_test_split

    def split_data(
        self,
        df: pd.DataFrame,
        label_column: str,
        test_size: float = 0.2,
        val_size: float = 0.25,
        random_state: float = 42,
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Splits data into training, validation, and test sets with stratified sampling.

        Parameters:
        - df: DataFrame containing the data.
        - label_column: Name of the column containing the labels.
        - test_size: Proportion of the dataset to include in the test split.
        - val_size: Proportion of the training dataset to include in the validation split.
        - random_state: Controls the shuffling applied to the data before applying the split.

        Returns:
        - X_train, X_val, X_test, y_train, y_val, y_test: Split datasets.
        """

        X = df.drop(label_column, axis=1)
        y = df[label_column]

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        val_size_adjusted = val_size / (1 - test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=random_state,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def one_hot_encode(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified columns of a DataFrame.

        Parameters:
        - df: DataFrame to be encoded.
        - columns: List of column names to encode.

        Returns:
        - DataFrame with one-hot encoded columns.
        """
        return pd.get_dummies(df, columns=columns, dtype=int)
