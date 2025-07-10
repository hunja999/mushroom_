import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# App title and description
st.title("🍄 Mushroom Edibility Predictor")
st.write("Select mushroom features to predict if it's edible or poisonous.")

# Load dataset
df = pd.read_csv("mushrooms.csv")

# Encode all columns
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Train the model (on 4 features only)
features = ['cap-shape', 'odor', 'gill-size', 'spore-print-color']
X = df[features]
y = df['class']

model = DecisionTreeClassifier()
model.fit(X, y)

# Feature encoding maps
cap_shape_display = {"Bell": 'b', "Conical": 'c', "Convex": 'x', "Flat": 'f', "Knobbed": 'k', "Sunken": 's'}
odor_display = {"Almond": 'a', "Anise": 'l', "Creosote": 'c', "Fishy": 'y', "Foul": 'f', "Musty": 'm', "None": 'n', "Pungent": 'p', "Spicy": 's'}
gill_size_display = {"Broad": 'b', "Narrow": 'n'}
spore_color_display = {"Black": 'k', "Brown": 'n', "Buff": 'b', "Chocolate": 'h', "Green": 'r', "Orange": 'o', "Purple": 'u', "White": 'w', "Yellow": 'y'}

# Model encoding maps (from LabelEncoder order)
cap_map = {'b': 0, 'c': 1, 's': 2, 'f': 3, 'k': 4, 'x': 5}
odor_map = {'a': 0, 'c': 1, 'f': 2, 's': 3, 'y': 4, 'p': 5, 'l': 6, 'm': 7, 'n': 8}
gill_map = {'b': 0, 'n': 1}
spore_map = {'b': 0, 'h': 1, 'k': 2, 'n': 3, 'o': 4, 'r': 5, 'u': 6, 'w': 7, 'y': 8}

# User input
cap_shape = st.selectbox("🧢 Cap Shape", list(cap_shape_display.keys()))
odor = st.selectbox("👃 Odor", list(odor_display.keys()))
gill_size = st.selectbox("🍃 Gill Size", list(gill_size_display.keys()))
spore_color = st.selectbox("🎨 Spore Print Color", list(spore_color_display.keys()))

if st.button("🔍 Predict"):
    input_data = [
        cap_map[cap_shape_display[cap_shape]],
        odor_map[odor_display[odor]],
        gill_map[gill_size_display[gill_size]],
        spore_map[spore_color_display[spore_color]],
    ]

    prediction = model.predict([input_data])[0]

    if prediction == 0:
        st.success("✅ This mushroom is **edible**.")
    else:
        st.error("☠️ This mushroom is **poisonous**.")


