import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data; fallback to raw if not found
num_cols = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
try:
    df = pd.read_csv('cleaned_penguins.csv')
except FileNotFoundError:
    url = 'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins_raw.csv'
    df = pd.read_csv(url)
    df['Species'] = df['Species'].str.split().str[0]
    df['Island'] = df['Island'].str.strip()
    df['Sex'] = df['Sex'].replace('.', np.nan).str.title()
    mask_no_useful = df[num_cols].isna().all(axis=1)
    df = df[~mask_no_useful]
    for col in num_cols:
        df[col] = df.groupby('Species')[col].transform(lambda x: x.fillna(x.mean()))

# Set a nice style
sns.set(style='whitegrid', palette='bright')

# Figure 1: Scatter plot of culmen length vs depth, colored by species, markers by sex
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Culmen Length (mm)', y='Culmen Depth (mm)', hue='Species', style='Sex', s=100)
plt.title('Bill Length vs Bill Depth by Species and Sex')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.legend(title='Species / Sex')
plt.savefig('exploratory_fig1.png')
plt.close()

# Figure 2: Boxplot of body mass by species and sex
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Species', y='Body Mass (g)', hue='Sex')
plt.title('Body Mass Distribution by Species and Sex')
plt.xlabel('Species')
plt.ylabel('Body Mass (g)')
plt.legend(title='Sex')
plt.savefig('exploratory_fig2.png')
plt.close()

# Figure 3: Pairplot of numerical measurements colored by species
num_df = df[['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Species']]
sns.pairplot(num_df, hue='Species', diag_kind='kde', markers=['o', 's', 'D'])
plt.suptitle('Pairwise Relationships Between Measurements by Species', y=1.02)
plt.savefig('exploratory_fig3.png')
plt.close()

# Figure 4: Correlation heatmap (numerical features only)
plt.figure(figsize=(8, 6))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Measurements')
plt.savefig('exploratory_fig4.png')
plt.close()

print("Exploratory figures saved as PNG files.")