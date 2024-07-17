import logging
from typing import AnyStr

import numpy as np
import rasterio
import matplotlib.pyplot as plt


class RasterOperations:
    """
    A class that provides operations for working with raster data.

    This class contains methods for flattening a multi-dimensional numpy array,
    converting a raster file to a numpy array, and masking the array with a
    no data value if present.

    Attributes:
        None
    """

    def read_raster_metadata(self, file_path):
        """
        Read a raster file and its metadata.

        Args:
            file_path (str): Path to the raster file.

        Returns:
            tuple: A tuple containing the numpy array of the raster data and its metadata.

        Metadata Dictionary Explanation:
        - 'driver': The format of the file (e.g., 'GTiff' for GeoTIFF).
        - 'dtype': The data type of the raster's pixels (e.g., 'float32').
        - 'nodata': The value used to represent missing data
                    (e.g., None if no specific nodata value is set).
        - 'width': The number of columns in the raster.
        - 'height': The number of rows in the raster.
        - 'count': The number of bands in the raster.
        - 'crs': The Coordinate Reference System (CRS) of the raster, defined by an EPSG code
                 (e.g., CRS.from_epsg(4326) for WGS84).
        - 'transform': An Affine transformation that defines the raster's georeferencing
                       (i.e., how pixel coordinates are mapped to geographic coordinates).
                       The parameters are coefficients of the affine transformation.
        """
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

            If the raster file has a defined no data value,
            the function will mask the array with the no data
            value and return the masked array.
            Otherwise, it will return the array as is.
        """
        with rasterio.open(file_path) as src:
            array = src.read()
            no_data_value = src.nodatavals[0]
            if no_data_value is None:
                return array
            array[array == no_data_value] = np.nan
            src.close()
            return array

    def flatten_array(self, array: np.ndarray, index=0):
        """
        Convert raster file to a 1D numpy array

        Args:
            array: multi-dimensional numpy array
            index: index of the band to be flattened

        Returns:
            flat_array: 1D numpy array
        """
        flat_array = array[index, :, :].flatten()
        return flat_array

    def compute_ndvi(
        self, nir_band: np.ndarray, red_band: np.ndarray
    ) -> np.ndarray | None:
        if nir_band.shape != red_band.shape:
            logging.error("NIR and Red bands must have the same shape.")
            return None

        np.seterr(divide="ignore", invalid="ignore")
        ndvi = (nir_band - red_band) / (nir_band + red_band)
        np.seterr(divide="warn", invalid="warn")

        return ndvi

    def visualize_array(self, array) -> None:
        """
        Visualize a 2D numpy array using matplotlib

        Args:
            array: 2D numpy array to be visualized
        """
        plt.figure(figsize=(10, 6))
        plt.imshow(array, cmap="viridis")
        plt.colorbar()
        plt.show()
        return None

    def apply_mask(self, data, mask, mask_value=np.nan) -> np.ndarray:
        """
        Apply a mask to the data array, replacing masked values
        with the specified mask value.

        Args:
            data (np.ndarray): The data array to be masked.
            mask (np.ndarray): A boolean array where True indicates
                               the position of values to be masked in the data array.
            mask_value (optional): The value to replace the masked values with.
                                   Default is np.nan.

        Returns:
            np.ndarray: The masked data array.
        """
        masked_data = np.copy(data)
        masked_data[mask] = mask_value
        return masked_data

    def export_raster(self, data, metadata, output_path) -> None:
        """
        Export data to a raster file with the given metadata.

        Args:
            data (np.ndarray): The raster data to be exported.
            metadata (dict): The metadata for the raster file
            output_path (str): The file path where the raster will be saved.
        """
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

    def add_band(self, raster_as_array, metadata, new_band) -> np.ndarray:
        """
        Adds a new band to the existing raster array.

        Args:
            raster_as_array (np.ndarray): The existing raster data array.
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
