import pandas as pd

df = pd.read_csv("extremism_data_final.csv")

df = df.drop_duplicates(subset=["Original_Message"])

df.to_csv("extremism_data_no_duplicates.csv", index = False)