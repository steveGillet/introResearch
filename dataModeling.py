import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the raw data from URL
url = 'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins_raw.csv'
df = pd.read_csv(url)

# Standardize and clean as before
df['Species'] = df['Species'].str.split().str[0]
df['Island'] = df['Island'].str.strip()
df['Sex'] = df['Sex'].replace('.', np.nan).str.title()
num_cols = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)']
mask_no_useful = df[num_cols].isna().all(axis=1)
df = df[~mask_no_useful]
for col in num_cols:
    df[col] = df.groupby('Species')[col].transform(lambda x: x.fillna(x.mean()))

# Drop rows with missing Sex for modeling
df = df.dropna(subset=['Sex'])

# Create combined target
df['Target'] = df['Species'] + ' ' + df['Sex']

# Features and target
X = df[['Culmen Length (mm)', 'Culmen Depth (mm)']]
y = df['Target']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Model (multinomial logistic regression)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Coefficients table
classes = le.classes_
coef_df = pd.DataFrame(model.coef_, index=classes, columns=['Culmen Length Coef', 'Culmen Depth Coef'])
coef_df['Intercept'] = model.intercept_

# Classification report
report = classification_report(y_test, y_pred, target_names=classes)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print("\nCoefficients:\n", coef_df)
print("\nClassification Report:\n", report)