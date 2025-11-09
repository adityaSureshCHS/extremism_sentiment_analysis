import pandas as pd

# Change this to your actual file name
input_file = "extremism_data.csv"
output_file = "extremism_data_final.csv"

# Read the CSV
df = pd.read_csv(input_file)

# Clean the second column: remove spaces + lowercase
df["Extremism_Label"] = (
    df["Extremism_Label"]
    .astype(str)              # ensure it's a string
    .str.replace(" ", "", regex=False)  # remove all spaces
    .str.upper()              # convert to lowercase
)

# Save to a new CSV
df.to_csv(output_file, index=False)

print("Done! Cleaned file saved as", output_file)