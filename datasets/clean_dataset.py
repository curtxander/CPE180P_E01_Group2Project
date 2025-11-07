import pandas as pd

# Load the CSV
df = pd.read_csv("raw_diamond_images/web_scraped/diamond_data.csv")

# Keep only the selected columns
df = df[["path_to_img", "shape", "colour", "cut"]]

# Save to a new CSV file
df.to_csv("diamond_data_filtered.csv", index=False)
