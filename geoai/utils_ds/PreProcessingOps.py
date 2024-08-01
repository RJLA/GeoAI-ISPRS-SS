from itertools import combinations
import numpy as np
from typing import Tuple
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class PreProcessingOperations:

    def scale_to_minmax(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This function scales the data to min-max scale.

        Args:
            X_train: pd.DataFrame: Training data
            X_test: pd.DataFrame: Test data

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Scaled data
        """
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

        return X_train_scaled, X_test_scaled

    def compute_pca_variance(
        self,
        X_train: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function computes PCA variance.

        Args:
            X_train: pd.DataFrame: Training data
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
        lda = LDA()
        lda.fit(X_train, y_train)

        explained_variance_ratio = lda.explained_variance_ratio_
        cumulative_discriminability = np.cumsum(explained_variance_ratio)

        return explained_variance_ratio, cumulative_discriminability

    def apply_pca(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        n_components: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This function applies PCA.

        Args:
            X_train: pd.DataFrame: Training data
            X_test: pd.DataFrame: Test data
            n_components: int: Number of components

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: PCA data
        """
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        X_train_pca = pd.DataFrame(
            X_train_pca, columns=[f"PC_{i}" for i in range(n_components)]
        )
        X_test_pca = pd.DataFrame(
            X_test_pca, columns=[f"PC_{i}" for i in range(n_components)]
        )

        return X_train_pca, X_test_pca

    def apply_lda(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        n_components: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies Linear Discriminant Analysis (LDA) on the given datasets.

        Args:
            X_train (pd.DataFrame): The training dataset.
            y_train (pd.DataFrame): The training labels.
            X_test (pd.DataFrame): The test dataset.
            n_components (int): The number of components to keep.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
            A tuple containing the transformed datasets for training,
            validation, and testing.
        """

        lda = LDA(n_components=n_components)
        X_train_lda = lda.fit_transform(X_train, y_train)
        X_test_lda = lda.transform(X_test)

        X_train_lda = pd.DataFrame(
            X_train_lda, columns=[f"LD_{i}" for i in range(n_components)]
        )
        X_test_lda = pd.DataFrame(
            X_test_lda, columns=[f"LD_{i}" for i in range(n_components)]
        )

        return X_train_lda, X_test_lda

    def polynomial_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        numerical_columns: list,
        degree: int = 2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies polynomial transformation to the specified columns of the DataFrame(s).

        Args:
            X_train (pd.DataFrame): Training DataFrame to be transformed.
            X_test (pd.DataFrame, optional): Test DataFrame to be transformed.
            numerical_columns (list): List of numerical_columns names to transform.
            degree (int): Degree of the polynomial transformation.

        Returns:
            tuple: Transformed DataFrames (X_train, X_test) with
                   original and polynomial transformed columns.
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
        X_test_transformed = transform_df(X_test, poly, numerical_columns)

        return X_train_transformed, X_test_transformed

    def make_one_hot_encoder(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        target_column: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encodes a specified column using one-hot encoding across training, validation, and test DataFrames.

        Args:
            X_train (pd.DataFrame): Training DataFrame.
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

        encoder = OneHotEncoder(dtype=int, sparse_output=False)
        X_train_encoded = encode_df(X_train, encoder, fit=True)
        X_test_encoded = encode_df(X_test, encoder)

        return X_train_encoded, X_test_encoded

    def make_ordinal_encoder(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        target_column: str,
        categories,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply ordinal encoding to the target column in the given DataFrames.

        Args:
            X_train (pd.DataFrame): Training DataFrame.
            X_test (pd.DataFrame): Test DataFrame.
            target_column (str): Name of the target column to encode.
            categories (list[list]): List of category lists for each feature.

        Returns:
            tuple: A tuple containing the modified training and test DataFrames.
        """
        encoder = OrdinalEncoder(categories=categories, dtype=int)
        X_train[f"{target_column}_encoded"] = encoder.fit_transform(
            X_train[[target_column]]
        )
        X_test[f"{target_column}_encoded"] = encoder.transform(X_test[[target_column]])
        del X_train[target_column]
        del X_test[target_column]

        return X_train, X_test

    def make_label_encoder(
        self, y_train: pd.Series, y_test: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Encodes target labels using label encoding across training, validation, and test sets.

        Args:
            y_train (pd.Series): Target labels for the training set.
            y_test (pd.Series): Target labels for the test set.

        Returns:
            tuple: A tuple containing the modified training, validation, and test Series.
        """
        le = LabelEncoder()
        y_train_encoded = pd.Series(
            le.fit_transform(y_train), index=y_train.index, name=y_train.name
        )
        y_test_encoded = pd.Series(
            le.transform(y_test), index=y_test.index, name=y_test.name
        )
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(label_mapping)

        return y_train_encoded, y_test_encoded

    def binarize_or_discretize(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        input_column: str,
        output_column: str,
        bin_edges: list,
        bin_labels: list,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create bins or discretize a specified column of a DataFrame based on provided bin edges and labels.

        Args:
            X_train (pd.DataFrame): Training DataFrame.
            X_test (pd.DataFrame): Test DataFrame.
            input_column (str): Name of the column to apply binning.
            output_column (str): Name of the column as a result of binning.
            bin_edges (list): List of edges to define the bins. Must be one more than the number of labels.
            bin_labels (list): List of labels for the bins.

        Returns:
            tuple: A tuple containing the modified training, validation, and test DataFrames.
        """

        def discretize(df):
            column_data = df[input_column].values
            df[output_column] = pd.cut(
                column_data, bins=bin_edges, labels=bin_labels, include_lowest=True
            )
            return df

        X_train = discretize(X_train)
        X_test = discretize(X_test)

        return X_train, X_test

    def interaction(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        columns: list,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Automatically generates column pairs from the passed list and computes
        addition, subtraction, multiplication, and division for each pair.

        Args:
            X_train (pd.DataFrame): Training DataFrame.
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
        X_test = apply_interactions(X_test, columns)

        return X_train, X_test
