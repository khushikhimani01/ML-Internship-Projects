import pandas as pd
import glob
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

files = glob.glob(os.path.join(DATA_DIR, "*.pkl"))
print("\nTotal daily files found:", len(files))

if not files:
    print("❌ No data files found. Make sure .pkl files are inside /data/")
    exit()

print("Example file:", files[0])

df = pd.read_pickle(files[0])

print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

if "TX_FRAUD" in df.columns:
    print("\nFraud distribution:")
    print(df["TX_FRAUD"].value_counts())
else:
    print("\n⚠️ TX_FRAUD column not found!")
