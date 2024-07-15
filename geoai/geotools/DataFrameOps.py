from ydata_profiling import ProfileReport
import pandas as pd
from numpy import ndarray


class DFOperations:
    """
    A class that provides operations for converting
    raster arrays to pandas DataFrames.
    """

    def convert_to_df(self, array: ndarray, column_name: str) -> pd.DataFrame:
        """
        Convert a raster array to a pandas DataFrame

        Args:
            array (np.ndarray): The input raster array to be converted.
            column_name (str): The name to be used for the DataFrame's column.

        Returns:
            df (DataFrame): The DataFrame containing the raster data.
        """
        print(f"Converting raster array to DataFrame with column name {column_name}")
        df = pd.DataFrame(array, columns=[column_name])
        return df

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
