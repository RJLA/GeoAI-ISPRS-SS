"""
This module contains various preprocessing operations for data manipulation and
transformation.

Classes:
    PreProcessingOperations: A class that encapsulates various preprocessing
    methods.
"""

__author__ = "Reginald Jay L. Argamosa"
__version__ = "0.1.0"
__email__ = "regi.argamosa@gmail.com"
__license__ = (
    "Reginald Jay L. Argamosa Personal Use License: See LICENSE file for details"
)

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
)


class PreProcessingOperations:
    """
    A class that encapsulates various preprocessing methods.

    Attributes:
        None

    Methods:
        scale_to_minmax: Scales the data to 0-1.
        compute_pca_variance: Computes PCA variance.
        compute_discriminability_ratios: Computes discriminability ratios.
        apply_pca: Applies PCA to the data.
        apply_lda: Applies Linear Discriminant Analysis (LDA) to the data.
        polynomial_transform: Applies polynomial transformation to the data.
        make_one_hot_encoder: Encodes a specified column using one-hot encoding.
        make_ordinal_encoder: Encodes a specified column using ordinal encoding.
        make_label_encoder: Encodes target labels using label encoding.
        binarize_or_categorize: Bins or categorizes a specified column of a DataFrame.
        interaction: Automatically generates column pairs and computes interactions.
    """

    def scale_to_minmax(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """
        This function scales the data to 0-1.

        Args:
            X_train (pd.DataFrame): Training data
            X_test (pd.DataFrame | None): Test data

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Scaled data
        """
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

            return X_train_scaled, X_test_scaled

        return X_train_scaled

    def compute_pca_variance(
        self,
        X_train: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function computes PCA variance.

        Args:
            X_train (pd.DataFrame): Training data
            n_components: int: Number of components

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: PCA variance
        """
        pca = PCA()
        pca.fit(X_train)
        var_exp = pca.explained_variance_ratio_
        cum_var_exp = np.cumsum(var_exp)
        return var_exp, cum_var_exp

    def compute_discriminability_ratios(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function computes discriminability ratios.

        Args:
            X_train: pd.DataFrame: Training data

        Returns:
            Tuple[np.ndarray, np.ndarray]: Explained variance ratio,
            Cumulative discriminability
        """
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train, y_train)

        explained_variance_ratio = lda.explained_variance_ratio_
        cumulative_discriminability = np.cumsum(explained_variance_ratio)

        return explained_variance_ratio, cumulative_discriminability

    def apply_pca(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame | None = None,
        n_components: int = 2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """
        This function applies PCA.

        Args:
            X_train (pd.DataFrame): Training data
            X_test (pd.DataFrame | None): Test data
            n_components: int: Number of components

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame: Transformed data to PCA
        """
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_train_pca = pd.DataFrame(
            X_train_pca, columns=[f"PC_{i}" for i in range(n_components)]
        )

        if X_test is not None:
            X_test_pca = pca.transform(X_test)
            X_test_pca = pd.DataFrame(
                X_test_pca, columns=[f"PC_{i}" for i in range(n_components)]
            )
            return X_train_pca, X_test_pca
        return X_train_pca

    def apply_lda(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame | None = None,
        n_components: int = 2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies Linear Discriminant Analysis (LDA) on the given datasets.

        Args:
            X_train (pd.DataFrame): The training dataset.
            y_train (pd.DataFrame): The training labels.
            X_test (pd.DataFrame | None): The test dataset.
            n_components (int): The number of components to keep.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame: Transformed data
            to LDA
        """

        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_train_lda = pd.DataFrame(
            X_train_lda, columns=[f"LD_{i}" for i in range(n_components)]
        )

        if X_test is not None:
            X_test_lda = lda.transform(X_test)
            X_test_lda = pd.DataFrame(
                X_test_lda, columns=[f"LD_{i}" for i in range(n_components)]
            )
            return X_train_lda, X_test_lda
        return X_train_lda

    def polynomial_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame | None = None,
        numerical_columns: list = [],
        degree: int = 2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """
        Applies polynomial transformation to the specified columns of the
        DataFrame(s).

        Args:
            X_train (pd.DataFrame): Training DataFrame to be transformed.
            X_test (pd.DataFrame | None): Test DataFrame to be transformed.
            numerical_columns (list): List of numerical_columns names to transform.
            degree (int): Degree of the polynomial transformation.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame: Polynomial
            transformed DataFrames.
        """
        if not numerical_columns or degree < 1:
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

            X_excluded = X.drop(columns=columns)

            return pd.concat([X_excluded, transformed_df], axis=1)

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_transformed = transform_df(X_train, poly, numerical_columns, fit=True)

        if X_test is not None:
            X_test_transformed = transform_df(X_test, poly, numerical_columns)

            return X_train_transformed, X_test_transformed

        return X_train_transformed

    def make_one_hot_encoder(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame | None = None,
        target_column: str = "",
    ) -> Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """
        Encodes a specified column using one-hot encoding across training,
        validation, and test DataFrames.

        Args:
            X_train (pd.DataFrame): Training DataFrame.
            X_test (pd.DataFrame, optional): Test DataFrame.
            target_column (str): The name of the column to be one-hot encoded.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame: One-hot encoded
            DataFrames.

        """

        def encode_df(X, encoder=None, fit=False):
            if X is None:
                return None
            if fit:
                encoder.fit(X[[target_column]])
            encoded = encoder.transform(X[[target_column]])
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out([target_column]),
                index=X.index,
            )
            X = pd.concat([X.drop(columns=[target_column]), encoded_df], axis=1)
            return X

        encoder = OneHotEncoder(dtype=int, sparse_output=False)
        X_train_encoded = encode_df(X_train, encoder, fit=True)

        if X_test is not None:
            X_test_encoded = encode_df(X_test, encoder)

            return X_train_encoded, X_test_encoded

        return X_train_encoded

    def make_ordinal_encoder(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame | None = None,
        target_column: str = "",
        categories=None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """
        Apply ordinal encoding to the target column in the given DataFrames.

        Args:
            X_train (pd.DataFrame): Training DataFrame.
            X_test (pd.DataFrame): Test DataFrame.
            target_column (str): Name of the target column to encode.
            categories (list[list]): List of category lists for each feature.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | pd.DataFrame: Modified
            DataFrames
        """
        encoder = OrdinalEncoder(categories=categories, dtype=int)
        X_train[f"{target_column}_encoded"] = encoder.fit_transform(
            X_train[[target_column]]
        )

        if X_test is not None:
            X_test[f"{target_column}_encoded"] = encoder.transform(
                X_test[[target_column]]
            )
            del X_train[target_column]
            del X_test[target_column]

            return X_train, X_test

        del X_train[target_column]
        return X_train

    def make_label_encoder(
        self, y_train: pd.Series, y_test: pd.Series | None = None
    ) -> Tuple[pd.Series, pd.Series] | pd.Series:
        """
        Encodes target labels using label encoding across training, validation,
        and test sets.

        Args:
            y_train (pd.Series): Target labels for the training set.
            y_test (pd.Series): Target labels for the test set.

        Returns:
            Tuple[pd.Series, pd.Series] | pd.Series: Encoded target labels.
        """
        le = LabelEncoder()
        y_train_encoded = pd.Series(
            le.fit_transform(y_train), index=y_train.index, name=y_train.name
        )
        class_mapping = {class_: index for index, class_ in enumerate(le.classes_)}
        print(class_mapping)

        if y_test is not None:
            y_test_encoded = pd.Series(
                le.transform(y_test), index=y_test.index, name=y_test.name
            )
            return y_train_encoded, y_test_encoded

        return y_train_encoded

    def binarize_or_categorize(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame | None = None,
        input_column: str = "",
        output_column: str = "",
        bin_edges: list = [],
        bin_labels: list = [],
    ) -> Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """
        Create bins or categorize a specified column of a DataFrame based on
        provided bin edges and labels.

        Args:
            X_train (pd.DataFrame): Training DataFrame.
            X_test (pd.DataFrame): Test DataFrame.
            input_column (str): Name of the column to apply binning.
            output_column (str): Name of the column as a result of binning.
            bin_edges (list): List of edges to define the bins. Must be one more
            than the number of labels.
            bin_labels (list): List of labels for the bins.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame: Binned or
            categorized DataFrames.
        """

        def categorize(df):
            column_data = df[input_column].values
            df[output_column] = pd.cut(
                column_data, bins=bin_edges, labels=bin_labels, include_lowest=True
            )
            return df

        X_train = categorize(X_train)

        if X_test is not None:
            X_test = categorize(X_test)

            return X_train, X_test

        return X_train

    def log_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame | None = None,
        columns: list = [],
    ) -> Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """
        Apply log transformation to the specified columns of the DataFrame(s).

        Args:
            X_train (pd.DataFrame): Training DataFrame to be transformed.
            X_test (pd.DataFrame | None): Test DataFrame to be transformed.
            columns (list): List of column names to transform.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame: Log-transformed
            DataFrames.
        """

        if not columns:
            raise ValueError("Columns list cannot be empty.")
        log_trasformer = FunctionTransformer(func=np.log1p)
        X_train_transformed = log_trasformer.fit_transform(X_train[columns])
        X_train_transformed = pd.DataFrame(
            X_train_transformed, columns=columns, index=X_train.index
        )

        if X_test is not None:
            X_test_transformed = log_trasformer.transform(X_test[columns])
            X_test_transformed = pd.DataFrame(
                X_test_transformed, columns=columns, index=X_test.index
            )

            return X_train_transformed, X_test_transformed

        return X_train_transformed
