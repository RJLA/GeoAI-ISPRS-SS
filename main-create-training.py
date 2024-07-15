# %%
# IMPORT MODULES
import glob
import os

import pandas as pd

from geoai.geotools.RasterOps import RasterOperations
from geoai.geotools.DataFrameOps import DFOperations
from geoai.geotools.VectorOps import VectorOperations

# %%
# SET GLOBAL VAR
RASTER_PATH = r"output_raster\rgbnirndvi.tif"
N_BANDS = 5
BAND_NAMES = [
    "BLUE",
    "GREEN",
    "RED",
    "NIR",
    "NDVI",
]

# %%
# INSTANTIATE CLASSES
raster_ops = RasterOperations()
df_ops = DFOperations()
vector_ops = VectorOperations()

# %%
# CLIP ROI SHAPEFILE TO RASTER
for shapefile in glob.glob(os.path.join("shapefiles", "*.shp")):
    roi_name = os.path.splitext(os.path.basename(shapefile))[0]
    output_raster_path = os.path.join("output_raster", f"{roi_name}.tif")
    vector_ops.clip_raster_with_shapefile(RASTER_PATH, shapefile, output_raster_path)


# %%
# # MAKE DF ON EVERY ROI RASTER
for roi_path in glob.glob(os.path.join("output_raster", "*.tif")):
    file_name = os.path.splitext(os.path.basename(roi_path))[0]
    if file_name in ["builtup", "trees", "water"]:
        df_bands = []
        array = raster_ops.raster_to_array(roi_path)
        for band_index, band_name in zip(range(N_BANDS), BAND_NAMES):
            flat_array = raster_ops.flatten_array(array, band_index)
            df = df_ops.convert_to_df(flat_array, band_name)
            df_bands.append(df)
        final_df_per_bands = pd.concat(df_bands, axis=1)
        final_df_per_bands.to_csv(f"csv_files/{file_name}.csv", index=False)

# %%
# concat the files
df_roi = []
for file_path in glob.glob(os.path.join("csv_files", "*.csv")):
    file_name = os.path.basename(file_path)
    if file_name != "training_data.csv":
        df = pd.read_csv(file_path)
        df = df.loc[~(df == 0).all(axis=1)]
        df["Class"] = os.path.splitext(file_name)[0]
        df_roi.append(df)
final_training = pd.concat(df_roi, axis=0)
final_training.to_csv("csv_files/training_data.csv", index=False)

# %%
