from itertools import combinations
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.preprocessing import OrdinalEncoder


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

    def split_data(
        self,
        df: pd.DataFrame,
        label_column: str,
        test_size: float = 0.2,
        val_size: float = 0.25,
        random_state: int = 1,
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """
        Splits data into training, validation, and test sets with stratified sampling.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            label_column (str): Name of the column containing the labels.
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the training dataset to include in the validation split.
            random_state (int): Controls the shuffling applied to the data before applying the split.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test: Split datasets.
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

    def polynomial_transform(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
        columns: list = None,
        degree: int = 2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Applies polynomial transformation to the specified columns of the DataFrame(s).

        Args:
            X_train (pd.DataFrame): Training DataFrame to be transformed.
            X_val (pd.DataFrame, optional): Validation DataFrame to be transformed.
            X_test (pd.DataFrame, optional): Test DataFrame to be transformed.
            columns (list): List of column names to transform.
            degree (int): Degree of the polynomial transformation.

        Returns:
            tuple: Transformed DataFrames (X_train, X_val, X_test) with
                   original and polynomial transformed columns.
        """
        if not columns or degree < 1:
            raise ValueError(
                "Columns list cannot be empty and degree must be at least 1."
            )

        def transform_df(X, poly, columns, fit=False):
            if X is None:
                return None
            features_to_transform = X[columns]
            transformed = (
                poly.fit_transform(features_to_transform)
                if fit
                else poly.transform(features_to_transform)
            )
            new_feature_names = poly.get_feature_names_out(columns)
            transformed_df = pd.DataFrame(
                transformed, columns=new_feature_names, index=X.index
            )
            return pd.concat([X, transformed_df], axis=1)

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_transformed = transform_df(X_train, poly, columns, fit=True)
        X_val_transformed = transform_df(X_val, poly, columns)
        X_test_transformed = transform_df(X_test, poly, columns)

        return X_train_transformed, X_val_transformed, X_test_transformed

    def make_one_hot_encoder(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        target_column: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Encodes a specified column using one-hot encoding across training, validation, and test DataFrames.

        Args:
            X_train (pd.DataFrame): Training DataFrame.
            X_val (pd.DataFrame, optional): Validation DataFrame.
            X_test (pd.DataFrame, optional): Test DataFrame.
            target_column (str): The name of the column to be one-hot encoded.

        Returns:
            tuple: A tuple containing the DataFrames with the target column one-hot encoded.
        """

        def encode_df(X, encoder=None, fit=False):
            if X is None:
                return None
            if fit:
                encoder.fit(X[[target_column]])
            encoded = encoder.transform(X[[target_column]]).toarray()
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out([target_column]),
                index=X.index,
            )
            X = pd.concat([X.drop(columns=[target_column]), encoded_df], axis=1)
            return X

        encoder = OneHotEncoder(dtype=int, sparse=False)
        X_train_encoded = encode_df(X_train, encoder, fit=True)
        X_val_encoded = encode_df(X_val, encoder) if X_val is not None else None
        X_test_encoded = encode_df(X_test, encoder) if X_test is not None else None

        return X_train_encoded, X_val_encoded, X_test_encoded

    def make_ordinal_encoder(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        target_column: str,
        categories: list[list],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply ordinal encoding to the target column in the given DataFrames.

        Args:
            X_train (pd.DataFrame): Training DataFrame.
            X_val (pd.DataFrame): Validation DataFrame.
            X_test (pd.DataFrame): Test DataFrame.
            target_column (str): Name of the target column to encode.
            categories (list[list]): List of category lists for each feature.

        Returns:
            tuple: A tuple containing the modified training,
            validation, and test DataFrames.
        """
        encoder = OrdinalEncoder(categories=categories)
        X_train[f"{target_column}_encoded"] = encoder.fit_transform(
            X_train[[target_column]]
        )
        X_val[f"{target_column}_encoded"] = encoder.transform(X_val[[target_column]])
        X_test[f"{target_column}_encoded"] = encoder.transform(X_test[[target_column]])
        del X_train[target_column]
        del X_val[target_column]
        del X_test[target_column]

        return X_train, X_val, X_test

    def make_label_encoder(
        self, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Encodes target labels using label encoding across training, validation, and test sets.

        Args:
            y_train (pd.Series): Target labels for the training set.
            y_val (pd.Series): Target labels for the validation set.
            y_test (pd.Series): Target labels for the test set.

        Returns:
            tuple: A tuple containing the modified training, validation, and test Series.
        """
        le = LabelEncoder()
        y_train_encoded = pd.Series(
            le.fit_transform(y_train), index=y_train.index, name=y_train.name
        )
        y_val_encoded = pd.Series(
            le.transform(y_val), index=y_val.index, name=y_val.name
        )
        y_test_encoded = pd.Series(
            le.transform(y_test), index=y_test.index, name=y_test.name
        )
        return y_train_encoded, y_val_encoded, y_test_encoded

    def binarize_or_discretize(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        input_column: str,
        output_column: str,
        bin_edges: list,
        bin_labels: list,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create bins or discretize a specified column of a DataFrame based on provided bin edges and labels.

        Args:
            X_train (pd.DataFrame): Training DataFrame.
            X_val (pd.DataFrame): Validation DataFrame.
            X_test (pd.DataFrame): Test DataFrame.
            input_column (str): Name of the column to apply binning.
            output_column (str): Name of the column as a result of binning.
            bin_edges (list): List of edges to define the bins. Must be one more than the number of labels.
            bin_labels (list): List of labels for the bins.

        Returns:
            tuple: A tuple containing the modified training, validation, and test DataFrames.
        """
        if len(bin_edges) != len(bin_labels) + 1:
            raise ValueError(
                "Number of bin edges must be one more than the number of bin labels."
            )
        X_train[output_column] = pd.cut(
            X_train[input_column],
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
        )
        X_val[output_column] = pd.cut(
            X_val[input_column], bins=bin_edges, labels=bin_labels, include_lowest=True
        )
        X_test[output_column] = pd.cut(
            X_test[input_column], bins=bin_edges, labels=bin_labels, include_lowest=True
        )

        return X_train, X_val, X_test

    def interaction(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        columns: list,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Automatically generates column pairs from the passed list and computes
        addition, subtraction, multiplication, and division for each pair.

        Args:
            X_train (pd.DataFrame): Training DataFrame.
            X_val (pd.DataFrame): Validation DataFrame.
            X_test (pd.DataFrame): Test DataFrame.
            columns (list): List of column names to compute interactions.

        Returns:
            tuple: A tuple containing the modified training, validation, and test DataFrames.
        """

        def apply_interactions(df, columns):
            for col1, col2 in combinations(columns, 2):
                df[f"{col1}+{col2}"] = df[col1] + df[col2]
                df[f"{col1}-{col2}"] = df[col1] - df[col2]
                df[f"{col1}*{col2}"] = df[col1] * df[col2]
                df[f"{col1}/{col2}"] = np.where(
                    df[col2] == 0, np.nan, df[col1] / df[col2]
                )
            return df

        X_train = apply_interactions(X_train, columns)
        X_val = apply_interactions(X_val, columns)
        X_test = apply_interactions(X_test, columns)

        return X_train, X_val, X_test
