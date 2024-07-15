# %%
import pandas as pd
from geoai.geotools.DataFrameOps import DFOperations

# %%
# OPEN CSV FILE
training_df = pd.read_csv("csv_files/training_data.csv")
print(training_df)
df_ops = DFOperations()
# %%
# GET THE DESCRIPTIVE STATS
df_ops.generate_profile_report(training_df, "EDA")
# %%
no_duplicates = df_ops.remove_duplicate_rows(training_df)
# %%
df_ops.generate_profile_report(no_duplicates, "eda_no_duplicates")
# %%
