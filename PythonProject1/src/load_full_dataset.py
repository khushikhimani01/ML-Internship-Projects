import pandas as pd
import glob
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

files = glob.glob(os.path.join(DATA_DIR, "*.pkl"))
print("Total files found:", len(files))

dfs = []
for f in files:
    df = pd.read_pickle(f)
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)
print("\nMerged dataset shape:", full_df.shape)

print("\nColumns:", full_df.columns.tolist())

print("\nFraud distribution:")
print(full_df["TX_FRAUD"].value_counts())
