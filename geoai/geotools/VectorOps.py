import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd


class VectorOperations:

    def clip_raster_with_shapefile(
        self, raster_path, shapefile_path, output_raster_path
    ):
        """
        Clips a raster file with a shapefile and saves the output.

        Parameters:
        raster_path (str): Path to the input raster file.
        shapefile_path (str): Path to the shapefile used for clipping.
        output_raster_path (str): Path where the clipped raster will be saved.
        """

        shapes = gpd.read_file(shapefile_path)
        if shapes.geometry.empty:
            raise ValueError("Shapefile has no geometries to clip with.")
        geometries = [mapping(shapes.geometry.unary_union)]

        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, geometries, crop=True)
            out_meta = src.meta

        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(out_image)
