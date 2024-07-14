# %%
import glob
import os

import pandas as pd

from geoai.raster_operations.RasterOps import RasterOperations
from geoai.raster_operations.DataFrameOps import DFOperations

# %%
RASTER_PATH = r"D:\Projects\GEOAI\data\roi"
N_BANDS = 4
BAND_NAMES = [
    "BLUE",
    "GREEN",
    "RED",
    "NIR",
]

# instantiate the classes
raster_ops = RasterOperations()
df_ops = DFOperations()


# iterate through the files
for file_path in glob.glob(os.path.join(RASTER_PATH, "*.tif")):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    df_bands = []
    array = raster_ops.raster_to_array(file_path)
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
        df.dropna(how="all", inplace=True)
        df["Class"] = os.path.splitext(file_name)[0]
        df_roi.append(df)
final_training = pd.concat(df_roi, axis=0)
final_training.to_csv("csv_files/training_data.csv", index=False)
