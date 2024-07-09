import logging
from typing import AnyStr

from numpy import nan
from numpy import ndarray

import rasterio

class RasterOperations:
    """
    A class that provides operations for working with raster data.

    This class contains methods for flattening a multi-dimensional numpy array,
    converting a raster file to a numpy array, and masking the array with a
    no data value if present.

    Attributes:
        None
    """

    def raster_to_array(self, file_path: AnyStr) -> ndarray:
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
        logging.info("Reading raster file - Converting to array: %s", file_path)
        with rasterio.open(file_path) as src:
            array = src.read()
            no_data_value = src.nodatavals[0]
            if no_data_value is None:
                return array
            array[array == no_data_value] = nan
            src.close()
            return array



    def flatten_array(self, array: ndarray, index=0):
        """
        Convert raster file to a 1D numpy array

        Args:
            array: multi-dimensional numpy array
            index: index of the band to be flattened

        Returns:
            flat_array: 1D numpy array
        """
        logging.info("Flattening raster array at index %s", index)
        flat_array = array[index, :, :].flatten()
        return flat_array


