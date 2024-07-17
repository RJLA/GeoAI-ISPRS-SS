from ydata_profiling import ProfileReport
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
