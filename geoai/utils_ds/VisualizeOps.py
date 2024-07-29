from sklearn.metrics import confusion_matrix
from ydata_profiling import ProfileReport
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class VisualizeOperations:
    """
    A class that provides various visualization operations for data analysis.

    Attributes:
        figure_size (tuple): The size of the figure for plotting.

    Methods:
        finalize_plot(): Adjusts the layout to be tight and displays the plot.
        generate_profile_report(df, title): Generates a profile report from a DataFrame and displays it as an iframe in a Notebook.
        plot_dual_scatter(df, x, y, hue): Create a scatter plot of two variables in a DataFrame.
        make_pairplot(df, hue): Create a pair plot to visualize the relationships between variables in a DataFrame.
        make_box_plot(df, title): Generate a box plot of values.
        make_3d_scatter(df, x, y, z, class_name): Create a 3D scatter plot using the specified DataFrame and column names.
        make_heatmap(df, title): Generate a heatmap of the correlation matrix for the given DataFrame.
        plot_row(row, class_label, title_prefix): Generate a line plot for the provided row with the DataFrame's column names as the x-axis.
        plot_spectral(df, grouping_column): Plots the spectral signature by class for a given DataFrame.
    """

    figure_size = (12, 8)

    @staticmethod
    def finalize_plot():
        """
        Adjusts the layout to be tight and displays the plot.
        """
        plt.tight_layout()
        plt.show()

    def generate_profile_report(
        self, df: pd.DataFrame, title="DataProfileReport"
    ) -> None:
        """
        Generates a profile report from a DataFrame and
        displays it as an iframe in a Notebook.

        Args:
            df (pd.DataFrame): The DataFrame to profile.
            title (str): The title of the profile report.

        Returns:
            None. Displays the profile
            report as an iframe in a Jupyter Notebook.
        """

        profile = ProfileReport(df, title=title)
        profile.to_notebook_iframe()
        profile.to_file(f"eda_reports/{title}.html")

    def plot_dual_scatter(self, df: pd.DataFrame, x: str, y: str, hue: str) -> None:
        """
        Create a scatter plot of two variables in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            x (str): The name of the column representing the x-axis variable.
            y (str): The name of the column representing the y-axis variable.
            hue (str): The column name in the DataFrame to be used for
                       coloring the plot.

        Returns:
            None
        """
        plt.figure(figsize=self.figure_size)
        sns.scatterplot(data=df, x=x, y=y, hue=hue)
        plt.title("Scatter Plot of NDVI vs NIR")
        self.finalize_plot()
        return None

    def make_pairplot(self, df: pd.DataFrame, hue: str) -> None:
        """
        Create a pair plot to visualize the relationships between
        variables in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the variables
                               to be plotted.
            hue (str): The column name in the DataFrame to be used for
                       coloring the plot.

        Returns:
            None
        """
        plt.figure(figsize=self.figure_size)
        sns.pairplot(df, hue=hue)
        self.finalize_plot()
        return None

    def make_box_plot(self, df: pd.DataFrame, title: str) -> None:
        """
        Generate a box plot of values.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            x (str): The name of the column representing the x-axis variable.
            y (str): The name of the column representing the y-axis variable.
            title (str): The title of the box plot.

        Returns:
            None
        """
        plt.figure(figsize=self.figure_size)
        df.plot(kind="box", vert=False)
        plt.title(title)
        self.finalize_plot()
        return None

    def make_3d_scatter(
        self, df: pd.DataFrame, x: str, y: str, z: str, class_name: str
    ) -> None:
        """
        Create a 3D scatter plot using the specified DataFrame and column names.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            x (str): The column name for the x-axis.
            y (str): The column name for the y-axis.
            z (str): The column name for the z-axis.
            class_name (str): The column name for the class.

        Returns:
            None
        """

        fig = plt.figure(figsize=self.figure_size)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            df[x],
            df[y],
            df[z],
            c=df[class_name].astype("category").cat.codes,
            label=df[class_name],
        )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        plt.title("3D Scatter Plot of {}, {}, {}".format(x, y, z))
        self.finalize_plot()

    def make_heatmap(self, df: pd.DataFrame, title: str) -> None:
        """
        Generate a heatmap of the correlation matrix for the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame for which the heatmap needs to be generated.

        Returns:
            None
        """
        numerical_df = df.select_dtypes(include=[np.number])
        plt.figure(figsize=self.figure_size)
        sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm")
        plt.title(title)
        self.finalize_plot()

    def plot_row(
        self, row: pd.Series, class_label: str, title_prefix: str = "Spectral Plot"
    ) -> None:
        """
        Generate a line plot for the provided row with the DataFrame's column names as the x-axis.

        Args:
            row (pd.Series): The row to be plotted.
            row_label (str or int): The label or index of the row being plotted.
            title_prefix (str): The prefix for the title of the plot.

        Returns:
            None
        """
        plt.figure(figsize=self.figure_size)
        plt.plot(row.index, row.values, marker="o", linestyle="-")
        plt.title(f"{title_prefix} for Class {class_label}")
        plt.xlabel("Columns")
        plt.ylabel("Values")
        plt.xticks(rotation=45)
        self.finalize_plot()

    def plot_spectral(self, df: pd.DataFrame, grouping_column: str) -> None:
        """
        Plots the spectral signature by class for a given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the spectral data.

        Returns:
            None
        """
        grouped = df.groupby(grouping_column)

        plt.figure(figsize=self.figure_size)

        for class_name, group in grouped:
            spectral_data = group.iloc[:, :-1]
            mean_spectral_values = spectral_data.mean()
            plt.plot(
                spectral_data.columns,
                mean_spectral_values,
                marker="o",
                linestyle="-",
                label=f"Class {class_name}",
            )

        plt.title("Spectral Signature by Class")
        plt.xlabel("Spectral Bands")
        plt.ylabel("Spectral Values")
        plt.xticks(rotation=45)
        plt.legend()
        self.finalize_plot()

    def plot_confusion_matrix(
        self,
        true_labels: list | np.ndarray,
        predicted_labels: list | np.ndarray,
        class_names: list | np.ndarray,
    ):
        """
        Plots a confusion matrix using matplotlib and seaborn.

        Args:
            true_labels (list or array): True labels of the data.
            predicted_labels (list or array): Predicted labels of the data.
            class_names (list of str, optional): Names of the classes for labeling the axes. Defaults to None.
        """
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=self.figure_size)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=(
                class_names
                if class_names is not None and len(class_names) > 0
                else "auto"
            ),
            yticklabels=(
                class_names
                if class_names is not None and len(class_names) > 0
                else "auto"
            ),
        )
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        self.finalize_plot()

    def plot_explained_variance_ratio(
        self, var_exp: np.ndarray, cum_var_exp: np.ndarray, n_columns: int
    ) -> None:
        """
        Plots the explained variance ratio for principal components.

        Args:
            var_exp (np.ndarray): Array of individual explained variance ratios.
            cum_var_exp (np.ndarray): Array of cumulative explained variance ratios.
            n_columns (int): Number of principal components.

        Returns:
            None
        """
        plt.figure(figsize=self.figure_size)
        plt.bar(
            range(1, n_columns),
            var_exp,
            align="center",
            label="Individual explained variance",
        )
        plt.step(
            range(1, n_columns),
            cum_var_exp,
            where="mid",
            label="Cumulative explained variance",
        )
        plt.ylabel("Explained variance ratio")
        plt.xlabel("Principal component index")
        plt.legend(loc="best")
        self.finalize_plot()

    def plot_discriminality_ratio(
        self, explained_variance_ratio, cumulative_discriminability
    ):
        """
        Plot the individual and cumulative discriminability ratios.

        Parameters:
        explained_variance_ratio (list): List of explained variance ratios.
        cumulative_discriminability (list): List of cumulative discriminability values.

        Returns:
        None
        """
        plt.figure(figsize=self.figure_size)
        plt.bar(
            range(1, len(explained_variance_ratio) + 1),
            explained_variance_ratio,
            alpha=0.6,
            align="center",
            label="Individual discriminability",
        )
        plt.step(
            range(1, len(cumulative_discriminability) + 1),
            cumulative_discriminability,
            where="mid",
            label="Cumulative discriminability",
            linestyle="-",
            marker="o",
        )
        plt.xlabel("Linear discriminants")
        plt.ylabel("Discriminability ratio")
        plt.ylim(0, 1.1)
        plt.legend(loc="best")
        plt.title("LDA Discriminability Ratios")
        self.finalize_plot()

    def plot_feature_importance(self, importance):
        """
        Plot the feature importance values.

        Args:
            feature_importance (list): DataFrame containing feature importance values.

        Returns:
            None
        """
        plt.figure(figsize=self.figure_size)
        plt.barh(importance["Feature"], importance["Importance"], align="center")
        plt.xlabel("Feature importance")
        plt.ylabel("Feature name")
        plt.title("Feature Importance")
        plt.rc("xtick", labelsize=10)
        plt.rc("ytick", labelsize=10)
        plt.gca().invert_yaxis()
        self.finalize_plot()

    def plot_learning_curve(
        self,
        train_sizes: np.ndarray,
        train_mean: np.ndarray,
        train_std: np.ndarray,
        test_mean: np.ndarray,
        test_std: np.ndarray,
        ylim: list = [0.5, 1.03],
    ) -> None:
        """
        Plots the learning curve for a machine learning model.

        Args:
            train_sizes (np.ndarray): Array of training sizes.
            train_mean (np.ndarray): Array of mean training accuracies.
            train_std (np.ndarray): Array of standard deviations of training accuracies.
            test_mean (np.ndarray): Array of mean validation accuracies.
            test_std (np.ndarray): Array of standard deviations of validation accuracies.
        """
        plt.figure(figsize=self.figure_size)
        plt.plot(
            train_sizes,
            train_mean,
            color="blue",
            marker="o",
            markersize=5,
            label="Training accuracy",
        )
        plt.fill_between(
            train_sizes,
            train_mean + train_std,
            train_mean - train_std,
            alpha=0.15,
            color="blue",
        )
        plt.plot(
            train_sizes,
            test_mean,
            color="green",
            linestyle="--",
            marker="s",
            markersize=5,
            label="Validation accuracy",
        )
        plt.fill_between(
            train_sizes,
            test_mean + test_std,
            test_mean - test_std,
            alpha=0.15,
            color="green",
        )
        plt.grid()
        plt.xlabel("Number of training examples")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.ylim(ylim)
        self.finalize_plot()

    def plot_validation_curve(
        self, param_range, train_mean, train_std, test_mean, test_std
    ) -> None:
        """
        Plots the validation curve for a given parameter range.

        Args:
            param_range (array-like): The range of parameter values.
            train_mean (array-like): The mean training accuracy for each parameter value.
            train_std (array-like): The standard deviation of the training accuracy for each parameter value.
            test_mean (array-like): The mean validation accuracy for each parameter value.
            test_std (array-like): The standard deviation of the validation accuracy for each parameter value.

        Returns:
            None
        """
        plt.figure(figsize=self.figure_size)
        plt.plot(
            param_range,
            train_mean,
            color="blue",
            marker="o",
            markersize=5,
            label="Training accuracy",
        )
        plt.fill_between(
            param_range,
            train_mean + train_std,
            train_mean - train_std,
            alpha=0.15,
            color="blue",
        )
        plt.plot(
            param_range,
            test_mean,
            color="green",
            linestyle="--",
            marker="s",
            markersize=5,
            label="Validation accuracy",
        )
        plt.fill_between(
            param_range,
            test_mean + test_std,
            test_mean - test_std,
            alpha=0.15,
            color="green",
        )
        plt.grid()
        plt.xscale("log")
        plt.legend(loc="lower right")
        plt.xlabel("Parameter C")
        plt.ylabel("Accuracy")
        plt.ylim([0.8, 1.0])
        self.finalize_plot()
