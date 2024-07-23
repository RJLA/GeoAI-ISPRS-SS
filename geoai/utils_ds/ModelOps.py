from typing import Tuple
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler


class PreProcessingOperations:

    def scale_to_minmax(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        This function scales the data to min-max scale.

        Args:
            X_train: pd.DataFrame: Training data
            X_val: pd.DataFrame: Validation data
            X_test: pd.DataFrame: Test data

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Scaled data
        """
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled


class AccuracyOperations:

    def calculate_classification_accuracy(
        self, y_true, y_pred
    ) -> Tuple[float, float, float, float]:
        """
        This function calculates classification metrics.

        Args:
            y_true: List: True labels
            y_pred: List: Predicted labels

        Returns:
            Tuple[float, float, float, float]: Accuracy, Precision, Recall, F1 Score
        """
        accuracy = accuracy_score(y_true, y_pred, normalize=True)
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        return accuracy, precision, recall, f1
