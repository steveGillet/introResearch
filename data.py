import pandas as pd
import numpy as np

# Load the raw data from URL
url = 'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins_raw.csv'
df = pd.read_csv(url)

# Standardize categorical variables
# Species: Extract common name (first word) for consistency
df['Species'] = df['Species'].str.split().str[0]

# Island: Strip any whitespace (already consistent, but good practice)
df['Island'] = df['Island'].str.strip()

# Sex: Replace '.' with NaN and standardize to title case
df['Sex'] = df['Sex'].replace('.', np.nan)
df['Sex'] = df['Sex'].str.title()  # Converts 'MALE' to 'Male', 'FEMALE' to 'Female'

# Identify key numerical columns
num_cols = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']

# Drop rows with no useful data (all key numericals missing)
mask_no_useful = df[num_cols].isna().all(axis=1)
df = df[~mask_no_useful]
print(f"Dropped {mask_no_useful.sum()} rows with no useful measurement data.")

# Impute any remaining missing numerical values with mean per species (though none after drop)
for col in num_cols:
    df[col] = df.groupby('Species')[col].transform(lambda x: x.fillna(x.mean()))

# Optionally impute isotopes if needed (mean per species)
isotope_cols = ['Delta 15 N (o/oo)', 'Delta 13 C (o/oo)']
for col in isotope_cols:
    df[col] = df.groupby('Species')[col].transform(lambda x: x.fillna(x.mean()))

# For sex: Leave as NaN to keep authenticity (11 missing). If you want to impute simply (e.g., mode per species):
# df['Sex'] = df.groupby('Species')['Sex'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))

# Optional: Convert Date Egg to datetime
df['Date Egg'] = pd.to_datetime(df['Date Egg'])

# Save cleaned CSV
df.to_csv('cleaned_penguins.csv', index=False)
print("Cleaned data saved to 'cleaned_penguins.csv'.")

# Quick summary after cleaning
print(df.info())
print(df.isna().sum())  # Should show 0 for numericals, 11 for Sex, some for Comments/Isotopes if not imputed
print(df['Species'].unique())
print(df['Island'].unique())
print(df['Sex'].unique())