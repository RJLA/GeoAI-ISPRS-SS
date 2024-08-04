"""
This module contains utility functions for working with raster data.

Classes:
    RasterOperations: A class that provides operations for reading, processing,
    and exporting raster data.
"""

__author__ = "Reginald Jay L. Argamosa"
__version__ = "0.1.0"
__email__ = "regi.argamosa@gmail.com"
__license__ = (
    "Reginald Jay L. Argamosa Personal Use License: See LICENSE file for details"
)

import logging
from typing import AnyStr

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


class RasterOperations:
    """
    A class that provides operations for reading, processing, and exporting
    raster data.

    Attributes:
        None

    Methods:
        read_raster_metadata: Read a raster file and its metadata.
        raster_to_array: Read raster file, mask no data, and convert it to a numpy array.
        flatten_array: Convert raster file to a 1D numpy array.
        compute_ndvi: Compute the Normalized Difference Vegetation Index (NDVI).
        visualize_array: Visualize a 2D numpy array using matplotlib.
        apply_mask: Apply a mask to the data array, replacing masked values with the specified mask value.
        export_raster: Export data to a raster file with the given metadata.
        add_band: Adds a new band to the existing raster array.
        column_to_raster: Writes a single-band raster to the specified output path.
        get_raster_dimensions: Get the dimensions of a raster file.
        compute_ndvi_using_df: Computes NDVI from NIR and Red bands in a DataFrame.
        compute_ndbi: Computes NDBI from NIR and SWIR bands in a DataFrame.
        compute_rei: Computes REI from NIR and BLUE bands.
        create_ndvi_bin: Create a binary NDVI mask from the NDVI band.
        create_ndvi_category: Create a categorical NDVI mask from the NDVI band.
    """

    def read_raster_metadata(self, file_path: AnyStr) -> dict:
        """
        Read a raster file and its metadata.

        Args:
            file_path (AnyStr): Path to the raster file.

        Returns:
            tuple: A tuple containing the numpy array of the raster data and its
            metadata.


        Metadata Dictionary Explanation:
        - 'driver': The format of the file (e.g., 'GTiff' for GeoTIFF).
        - 'dtype': The data type of the raster's pixels (e.g., 'float32').
        - 'nodata': The value used to represent missing data
                    (e.g., None if no specific nodata value is set).
        - 'width': The number of columns in the raster.
        - 'height': The number of rows in the raster.
        - 'count': The number of bands in the raster.
        - 'crs': The Coordinate Reference System (CRS) of the raster, defined by
          an EPSG code (e.g., CRS.from_epsg(4326) for WGS84).
        - 'transform': An Affine transformation that defines the raster's
          georeferencing (i.e., how pixel coordinates are mapped to geographic
          coordinates). The parameters are coefficients of the affine transformation.
        """
        logging.info(f"Reading raster metadata from {file_path}")
        with rasterio.open(file_path) as src:
            metadata = src.meta
        return metadata

    def raster_to_array(self, file_path: AnyStr) -> np.ndarray:
        """
        Read raster file, mask no data, and convert it to a numpy array.

        Args:
            file_path (AnyStr): Path to the raster file.

        Returns:
            ndarray: The resulting numpy array after conversion.
        """
        logging.info(f"Reading raster file from {file_path}")
        with rasterio.open(file_path) as src:
            array = src.read()
            no_data_value = src.nodatavals[0]
            if no_data_value is None:
                return array
            array[array == no_data_value] = np.nan
            src.close()
            return array

    def flatten_array(self, array: np.ndarray, index=0) -> np.ndarray:
        """
        Convert raster file to a 1D numpy array

        Args:
            array (np.ndarray): multi-dimensional numpy array
            index (int): index of the band to be flattened

        Returns:
            np.ndarray: 1D numpy array
        """
        logging.info("Flattening the raster array")
        flat_array = array[index, :, :].flatten()
        return flat_array

    def compute_ndvi(
        self, nir_band: np.ndarray, red_band: np.ndarray
    ) -> np.ndarray | None:
        """
        Compute the Normalized Difference Vegetation Index (NDVI).

        NDVI is calculated using the formula:
        NDVI = (NIR - Red) / (NIR + Red)

        Args:
            nir_band (np.ndarray): The Near-Infrared (NIR) band of the image.
            red_band (np.ndarray): The Red band of the image.

        Returns:
            np.ndarray | None: The computed NDVI values as a numpy array, or
            None if the input bands do not have the same shape.
        """
        logging.info("Computing NDVI")
        if nir_band.shape != red_band.shape:
            logging.error("NIR and Red bands must have the same shape.")
            return None

        np.seterr(divide="ignore", invalid="ignore")
        ndvi = (nir_band - red_band) / (nir_band + red_band)
        np.seterr(divide="warn", invalid="warn")

        return ndvi

    def visualize_array(self, array: np.ndarray, title: str) -> None:
        """
        Visualize a 2D numpy array using matplotlib

        Args:
            array (np.ndarray): 2D numpy array to be visualized
            title (str): title of the plot

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        plt.imshow(array, cmap="viridis")
        plt.title(title)
        plt.colorbar()
        plt.show()
        return None

    def apply_mask(
        self, data: np.ndarray, mask: np.ndarray, mask_value: int | float = np.nan
    ) -> np.ndarray:
        """
        Apply a mask to the data array, replacing masked values
        with the specified mask value.

        Args:
            data (np.ndarray): The data array to be masked.
            mask (np.ndarray): A boolean array where True indicates the position
            of values to be masked in the data array.
            mask_value (optional): The value to replace the masked values with.
            Default is np.nan.

        Returns:
            np.ndarray: The masked data array.
        """
        logging.info("Applying mask to the data array")
        masked_data = np.copy(data)
        masked_data[mask] = mask_value
        return masked_data

    def export_raster(
        self, data: np.ndarray, metadata: dict, output_path: AnyStr
    ) -> None:
        """
        Export data to a raster file with the given metadata.

        Args:
            data (np.ndarray): The raster data to be exported.
            metadata (dict): The metadata for the raster file
            output_path (str): The file path where the raster will be saved.

        Returns:
            None
        """
        logging.info(f"Exporting raster to {output_path}")
        if data.ndim == 2:
            metadata.update(count=1, height=data.shape[0], width=data.shape[1])
        elif data.ndim == 3:
            metadata.update(
                count=data.shape[0], height=data.shape[1], width=data.shape[2]
            )

        with rasterio.open(output_path, "w", **metadata) as dst:
            if data.ndim == 2:
                dst.write(data, 1)
                return None
            dst.write(data)
            return None

    def add_band(
        self, raster_as_array: np.ndarray, metadata: dict, new_band: np.ndarray
    ) -> np.ndarray:
        """
        Adds a new band to the existing raster array.

        Args:
            raster_as_array (np.ndarray): The existing raster data array.
            metadata (dict): The metadata of the existing raster.
            new_band (np.ndarray): The new band data to add.

        Returns:
            np.ndarray: The updated raster array with the new band added.
        """
        if raster_as_array.ndim == 2:
            raster_as_array = raster_as_array[np.newaxis, :, :]

        if new_band.ndim != 2 or new_band.shape != raster_as_array.shape[1:]:
            raise ValueError(
                "New band must be 2D and match the shape of the existing bands."
            )

        updated_array = np.concatenate(
            (raster_as_array, new_band[np.newaxis, :, :]), axis=0
        )

        metadata["count"] = updated_array.shape[0]

        return updated_array

    def column_to_raster(
        self,
        output_path: AnyStr,
        dataframe: pd.DataFrame,
        band_name: str,
        metadata: tuple,
        dtype="float32",
    ) -> None:
        """
        Writes a single-band raster to the specified output path.

        Args:
            output_path (str): the path to the output raster file.
            dataframe (DataFrame): the DataFrame containing the column to write.
            band_name (str): the name of the column in the DataFrame to write.
            metadata (tuple): containing (height, width, CRS, Affine transform).
            dtype (str): the data type of the output raster.

        Returns:
            None
        """
        height, width, crs, transform = metadata
        band = dataframe[band_name].values.reshape(height, width)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(band, 1)
            dst.close()
        logging.info(f"Raster written to {output_path}")
        return None

    def get_raster_dimensions(self, file_path: AnyStr) -> tuple:
        """
        Get the dimensions of a raster file

        Args:
            file_path (AnyStr): path to the raster file

        Returns:
            original_height (int): height of the raster file
            original_width (int): width of the raster file
            original_crs (CRS): crs of the raster file
            original_transform (Affine): affine transformation
            of the raster file

        """
        with rasterio.open(file_path) as src:
            original_height = src.height
            original_width = src.width
            original_crs = src.crs
            original_transform = src.transform
            src.close()

        return original_height, original_width, original_crs, original_transform

    def compute_ndvi_using_df(
        self, df: pd.DataFrame, nir_band: str, red_band: str
    ) -> pd.DataFrame:
        """
        Computes NDVI from NIR and Red bands in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the NIR and Red bands.
            nir_band (str): The name of the NIR band column.
            red_band (str): The name of the Red band column.

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'NDVI' column.
        """
        df["NDVI"] = (df[nir_band] - df[red_band]) / (df[nir_band] + df[red_band])
        return df

    def compute_ndbi(
        self, df: pd.DataFrame, nir_band: str, swir_band: str
    ) -> pd.DataFrame:
        """
        Computes NDBI from NIR and SWIR bands in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the NIR and SWIR bands.
            nir_band (str): The name of the NIR band column.
            swir_band (str): The name of the SWIR band column.

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'NDBI' column.
        """
        df["NDBI"] = (df[swir_band] - df[nir_band]) / (df[swir_band] + df[nir_band])
        return df

    def compute_rei(self, df: pd.DataFrame, nir_band: str, blue: str) -> pd.DataFrame:
        """
        Computes REI from NIR and BLUE bands.

        Args:
            df (pd.DataFrame): The DataFrame containing the NIR and SWIR bands.
            nir_band (str): The name of the NIR band column.
            blue (str): The name of the BLUE band column.

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'REI' column.
        """

        df["REI"] = (df[nir_band] - df[blue]) / (df[nir_band] + df[blue] * df[nir_band])
        return df

    def create_ndvi_bin(self, df: pd.DataFrame, ndvi_band: str) -> pd.DataFrame:
        """
        Create a binary NDVI mask from the NDVI band.

        Args:
            df (pd.DataFrame): The DataFrame containing the NDVI band.
            ndvi_band (str): The name of the NDVI band column.

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'NDVI_bin'
            column.

        """
        ndvi_binary_edges = [-float("inf"), 0.5, float("inf")]
        ndvi_binary_labels = ["non_veg", "veg"]

        column_data = df[ndvi_band].values
        df["NDVI_binary"] = pd.cut(
            column_data,
            bins=ndvi_binary_edges,
            labels=ndvi_binary_labels,
            include_lowest=True,
        )

        return df

    def create_ndvi_category(self, df: pd.DataFrame, ndvi_band: str) -> pd.DataFrame:
        """
        Create a categorical NDVI mask from the NDVI band.

        Args:
            df (pd.DataFrame): The DataFrame containing the NDVI band.
            ndvi_band (str): The name of the NDVI band column.

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'NDVI_cat' column.
        """
        ndvi_category_edges = [-float("inf"), 0.2, 0.5, float("inf")]
        ndvi_category_labels = ["low_veg", "medium_veg", "high_veg"]

        column_data = df[ndvi_band].values
        df["NDVI_categorized"] = pd.cut(
            column_data,
            bins=ndvi_category_edges,
            labels=ndvi_category_labels,
            include_lowest=True,
        )

        return df

    def indices_binary_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute NDVI, NDBI, and REI from the specified bands in the DataFrame.
        Make binary and categoriy for NDVI.

        Args:
            df (pd.DataFrame): The DataFrame containing the bands.

        Returns:
            pd.DataFrame: The input DataFrame with additional columns for NDVI,
            NDBI, REI, Binary and Category.
        """

        df = self.compute_ndvi_using_df(df, "NIR", "RED")
        df = self.compute_ndbi(df, "NIR", "SWIR")
        df = self.compute_rei(df, "NIR", "BLUE")

        df = self.create_ndvi_category(df, "NDVI")
        df = self.create_ndvi_bin(df, "NDVI")

        return df
