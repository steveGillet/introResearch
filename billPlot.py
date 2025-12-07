import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and clean (as before)
url = 'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins_raw.csv'
df = pd.read_csv(url)
df['Species'] = df['Species'].str.split().str[0]
df['Sex'] = df['Sex'].replace('.', pd.NA).str.title()
num_cols = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
df = df[~df[num_cols].isna().all(axis=1)]
for col in num_cols:
    df[col] = df.groupby('Species')[col].transform(lambda x: x.fillna(x.mean()))
df = df.dropna(subset=['Sex'])  # For complete markers

# Publication style
sns.set_context('paper', font_scale=1.2)
sns.set_style('white')
palette = sns.color_palette('colorblind', 3)

plt.figure(figsize=(6, 4.5), dpi=300)
sns.scatterplot(data=df, x='Culmen Length (mm)', y='Culmen Depth (mm)', hue='Species', style='Sex', 
                palette=palette, s=80, edgecolor='k', linewidth=0.5, alpha=0.9, markers=['o', 's'])
plt.title('Bill Dimensions by Species and Sex', fontsize=14)
plt.xlabel('Bill Length (mm)', fontsize=12)
plt.ylabel('Bill Depth (mm)', fontsize=12)
plt.legend(title='Species / Sex', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('final_fig1.pdf', format='pdf', bbox_inches='tight')
plt.close()