import pandas as pd

# 1. Load & normalize columns
df = pd.read_csv(
    'data/violence_dataset.tdf',
    sep='\t',
    skipinitialspace=True,
    engine='python'
)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# 2. Count how many “Yes” vs “No” in the violence column
counts = df['violence'].value_counts()
print("Violence label distribution:")
print(counts)