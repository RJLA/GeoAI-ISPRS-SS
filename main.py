#%%
import glob
import os

import pandas as pd

from geoai.raster_operations.RasterOps import RasterOperations
from geoai.raster_operations.DataFrameOps import DFOperations

#%%
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
for files in glob.glob(os.path.join(RASTER_PATH, "*.tif")):
    file_name = os.path.basename(files[:-4])
    df_bands = []
    for band_index, band_name in zip(range(N_BANDS), BAND_NAMES):
        array = raster_ops.raster_to_array(files)
        flat_array = raster_ops.flatten_array(array, band_index)
        df = df_ops.convert_to_df(flat_array, band_name)
        df_bands.append(df)
    final_df_per_bands = pd.concat(df_bands, axis=1)
    final_df_per_bands.to_csv(f"csv_files/{file_name}.csv", index = False)

#%%



# #%%
# NUMBER_OF_BANDS = 4 # or raster_as_array.shape[0]
# BAND_NAMES = [
#     "BLUE",
#     "GREEN",
#     "RED",
#     "NIR",
# ]
# RASTER_PATH = r"D:\Projects\GEOAI\data\roi"
# array = raster_to_array(RASTER_PATH)
#  # empty list to store the dataframes
# DF_LIST = [] 
# # iterate through the bands
# for band_index, band_name in zip(range(NUMBER_OF_BANDS), BAND_NAMES):
#     flat_array = flatten_array(array, band_index)
#     df = convert_to_df(flat_array, band_name)
#     df_list.append(df)

# # concatenate the dataframes
# dataframe = pd.concat(df_list, axis=1)

# # convert raster to array
# raster_as_array = raster_to_array(raster_path)
# # %%

# %%
