# %%
from geoai.geotools.RasterOps import RasterOperations

# %%
# SET RASTER PATH, INSTANTIATE CLASS
raster_ops = RasterOperations()
RASTER_FILE_PATH = "s2_data/QC_S2.tif"

# %%
# READ RASTER METADATA
raster_metadata = raster_ops.read_raster_metadata(RASTER_FILE_PATH)


# %%
# READ RASTER AS ARRAY
raster_as_array = raster_ops.raster_to_array(RASTER_FILE_PATH)

# %%
# COMPUTE FOR NDVI
red = raster_as_array[0]
nir = raster_as_array[1]

ndvi = raster_ops.compute_ndvi(nir, red)
# %%
# VISUALIZE
raster_ops.visualize_array(ndvi)

# %%
# DEFINE THRESHOLD
ndvi_threshold = 0.2
# CREATE A MASK RASTER
vegetation_mask = ndvi < ndvi_threshold

# APPLY THE MASK
vegetation_ndvi = raster_ops.apply_mask(ndvi, vegetation_mask, mask_value=0)

# VISUALIZE THE MASKED DATA
raster_ops.visualize_array(vegetation_ndvi)
# %%
# EXPORT NDVI THRESHOLD
ndvi_thresh_path = "output_raster/ndvi_thresh.tif"
raster_ops.export_raster(vegetation_ndvi, raster_metadata, ndvi_thresh_path)

# %%
# EXPORT MULTI-BAND RASTER
rgbnirndvi_path = "output_raster/rgbnirndvi.tif"
rgbnirndvi = raster_ops.add_band(raster_as_array, raster_metadata, ndvi)
raster_ops.export_raster(rgbnirndvi, raster_metadata, rgbnirndvi_path)
# %%
