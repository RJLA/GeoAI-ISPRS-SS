# %%
import pandas as pd
from geoai.geotools.DataFrameOps import DFOperations

# %%
# OPEN CSV FILE
training_df = pd.read_csv("csv_files/training_data.csv")
print(training_df)
df_ops = DFOperations()
# %%
# # VISUALIZE DATASETS
df_ops.plot_dual_scatter(training_df, "NIR", "BLUE")
df_ops.make_pairplot(training_df, "Class")
df_ops.make_box_plot(training_df, "Boxplot")
df_ops.make_3d_scatter(training_df, "NIR", "BLUE", "NIR", "Class")
df_ops.make_heatmap(training_df, "Heatmap")
df_ops.plot_row(training_df.iloc[1, :-2], "builtup")
df_ops.plot_row(training_df.iloc[2885, :-2], "trees")
df_ops.plot_row(training_df.iloc[10625, :-2], "water")
df_ops.plot_spectral(training_df)

# %%
# VISUALIZE USING PROFILE
df_ops.generate_profile_report(training_df, "EDA")

# # %%
# REMOVE DUPLICATE ROWS
no_duplicates = df_ops.remove_duplicate_rows(training_df)
no_duplicates.to_csv("csv_files/final_training_df.csv", index=False)
