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
