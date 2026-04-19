import pandas as pd

df = pd.read_csv("data/cybersecurity.csv")

df_sample = df.sample(frac=0.15, random_state=42)

df_remaining = df.drop(df_sample.index)

df_sample.to_csv("SavedResults/Test.csv", index=False)
df_remaining.to_csv("SavedResults/TrainVal.csv", index=False)