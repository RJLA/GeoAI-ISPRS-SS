import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
)
import xgboost as xgb
import numpy as np
from typing import Tuple

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve,
)
from sklearn.model_selection import validation_curve


class ModelOperations:

    def select_features(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame
    ) -> Tuple[list, pd.DataFrame]:
        """
        Selects the important features from the given dataset
        using Gradient Boosting Classifier.

        Args:
            X_train (pd.DataFrame): The input training dataset.
            y_train (pd.DataFrame): The target training dataset.

        Returns:
            Tuple[list, pd.DataFrame]: A tuple containing the selected
            features and their importance.
        """
        X_train["noise_column"] = np.random.rand(len(X_train))
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
        feature_importances = model.feature_importances_
        feature_names = X_train.columns
        importance = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importances}
        )
        importance = importance.sort_values(by="Importance", ascending=False)

        importance_random_value = importance[importance["Feature"] == "noise_column"][
            "Importance"
        ].values[0]
        selected_features = list(
            importance[importance["Importance"] > importance_random_value]["Feature"]
        )

        return selected_features, importance

    def calculate_classification_accuracy(
        self, y_true, y_pred
    ) -> Tuple[float, float, float, float]:
        """
        This function calculates classification metrics.

        Args:
            y_true: List: True labels
            y_pred: List: Predicted labels

        Returns:
            Tuple[float, float, float, float]: Accuracy,
            Precision, Recall, F1 Score
        """
        accuracy = accuracy_score(y_true, y_pred, normalize=True)
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        return accuracy, precision, recall, f1

    def stratify_k_fold_cv(
        self, X_train: pd.DataFrame, y_train: pd.Series, n_splits: int, model
    ) -> Tuple[float, float]:
        """
        Perform stratified k-fold cross-validation on the given dataset.

        Args:
            X_train (pd.DataFrame): The input features for training.
            y_train (pd.Series): The target variable for training.
            n_splits (int): The number of folds to split the data into.
            model: The machine learning model to be trained and evaluated.

        Returns:
            Tuple[float, float]: A tuple containing the mean accuracy and
            standard deviation of the accuracy across all folds.
        """
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True).split(X_train, y_train)
        scores = []
        for k, (train_index, test_index) in enumerate(kfold):
            model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
            pred = model.predict(X_train.iloc[test_index])
            score = f1_score(y_train.iloc[test_index], pred, average="weighted")
            scores.append(score)
            print(f"Fold: {k+1:02d}, " f"Acc.: {score:.3f}")
        mean_acc = np.mean(scores)
        std_acc = np.std(scores)
        print(f"\nCV accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")

        return mean_acc, std_acc

    def compute_metrics_for_learning_curve(
        self, X_train: pd.DataFrame, y_train: pd.Series, model, cv: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the metrics for a learning curve.

        Args:
            X_train (pd.DataFrame): The input features for training.
            y_train (pd.Series): The target variable for training.
            model: The machine learning model to evaluate.
            cv (int): The number of cross-validation folds. Default is 10.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing the following arrays:
                - train_sizes: The number of training samples used in each fold.
                - train_mean: The mean training scores for each fold.
                - train_std: The standard deviation of the training scores for each fold.
                - test_mean: The mean test scores for each fold.
                - test_std: The standard deviation of the test scores for each fold.
        """
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model,
            X=X_train,
            y=y_train,
            train_sizes=np.linspace(0.1, 1.0, 100),
            cv=cv,
            n_jobs=-1,
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        return train_sizes, train_mean, train_std, test_mean, test_std

    def compute_metric_for_validation_curve(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_range: list,
        param_name: str,
        model,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the metrics for a validation curve.

        Args:
            X_train (pd.DataFrame): The input features for training.
            y_train (pd.Series): The target variable for training.
            param_range (list): The range of values for the hyperparameter.
            param_name (str): The name of the hyperparameter.
            model: The machine learning model to evaluate.

            Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """

        param_range = param_range
        train_scores, test_scores = validation_curve(
            estimator=model,
            X=X_train,
            y=y_train,
            param_name=param_name,
            param_range=param_range,
            cv=10,
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        return train_mean, train_std, test_mean, test_std

    def find_best_hyperparameters_gs(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model,
        param_grid: dict,
        accuracy: str,
    ) -> Tuple[dict, float]:
        """
        Find the best hyperparameters for a given model using GridSearchCV.

        Args:
            X_train (pd.DataFrame): The input features for training.
            y_train (pd.Series): The target variable for training.
            model: The machine learning model to evaluate.
            param_grid (dict): The hyperparameter grid to search over.

        Returns:
            Tuple[dict, float]: A tuple containing the best hyperparameters
            and the best score obtained.
        """
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=accuracy,
            cv=10,
            refit=True,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        return best_params, best_score

    def find_best_hyperparameters_rs(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model,
        param_grid: dict,
        accuracy: str,
    ) -> Tuple[dict, float]:
        """
        Find the best hyperparameters for a given model using RandomizedSearchCV.

        Args:
            X_train (pd.DataFrame): The input features for training.
            y_train (pd.Series): The target variable for training.
            model: The machine learning model to evaluate.
            param_grid (dict): The hyperparameter grid to search over.

        Returns:
            Tuple[dict, float]: A tuple containing the best hyperparameters
            and the best score obtained.
        """
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=accuracy,
            cv=10,
            refit=True,
            n_jobs=-1,
            n_iter=20,
            random_state=1,
        )
        random_search.fit(X_train, y_train)
        best_params = random_search.best_params_
        best_score = random_search.best_score_

        return best_params, best_score

    def make_pipeline(self, classifier) -> Pipeline:
        """
        Create a pipeline for the given classifier.

        Args:
            classifier: The machine learning classifier to use.

        Returns:
            Pipeline: The machine learning pipeline.
        """

        numerical_columns = [
            "BLUE",
            "GREEN",
            "RED",
            "NIR",
            "SWIR",
            "NDVI",
            "NDBI",
            "REI",
        ]
        one_hot_encoder_columns = ["NDVI_bin"]
        ordinal_encoder_columns = ["NDVI_dis"]
        categories = [["low_veg", "medium_veg", "high_veg"]]

        numerical_transformer = Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=2)),
                ("select", SelectKBest(k=24)),
            ]
        )
        one_hot_transformer = OneHotEncoder(dtype=int, sparse_output=False)
        ordinal_transformer = OrdinalEncoder(categories=categories, dtype=int)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_columns),
                ("onehot", one_hot_transformer, one_hot_encoder_columns),
                ("ordinal", ordinal_transformer, ordinal_encoder_columns),
            ],
            remainder="drop",
        )
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("scale", MinMaxScaler()),
                ("dim_reduce", LinearDiscriminantAnalysis(n_components=3)),
                ("classifier", classifier),
            ]
        )

        return pipeline
