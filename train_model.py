import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("mushrooms.csv")

# Encode all columns
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# selected 4 features
X = df[['cap-shape', 'odor', 'gill-size', 'spore-print-color']]
y = df['class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print(" Model trained and saved locally as model.pkl")
