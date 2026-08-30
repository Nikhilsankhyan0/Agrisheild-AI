import pandas as pd

df = pd.read_csv("agrishield_preprocessed.csv")

print("Shape:", df.shape)

print("\nColumns:")

print(df.columns.tolist())
