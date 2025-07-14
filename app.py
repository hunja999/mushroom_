import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# App title and description
st.title("🍄 Mushroom Edibility Predictor")
st.write("Select mushroom features to predict if it's edible or poisonous.")

# Load and preprocess dataset
df = pd.read_csv("mushrooms.csv")

# Only use selected features
features = ['cap-shape', 'odor', 'gill-size', 'spore-print-color']
X = df[features]
y = df['class']

# Encode selected features
label_encoders = {}
for col in features + ['class']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df[features]
y = df['class']

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Human-friendly options mapped to single-character values
cap_shape_display = {"Bell": 'b', "Conical": 'c', "Convex": 'x', "Flat": 'f', "Knobbed": 'k', "Sunken": 's'}
odor_display = {"Almond": 'a', "Anise": 'l', "Creosote": 'c', "Fishy": 'y', "Foul": 'f', "Musty": 'm', "None": 'n', "Pungent": 'p', "Spicy": 's'}
gill_size_display = {"Broad": 'b', "Narrow": 'n'}
spore_color_display = {"Black": 'k', "Brown": 'n', "Buff": 'b', "Chocolate": 'h', "Green": 'r', "Orange": 'o', "Purple": 'u', "White": 'w', "Yellow": 'y'}

# Get user input
cap_shape = st.selectbox("🧢 Cap Shape", list(cap_shape_display.keys()))
odor = st.selectbox("👃 Odor", list(odor_display.keys()))
gill_size = st.selectbox("🍃 Gill Size", list(gill_size_display.keys()))
spore_color = st.selectbox("🎨 Spore Print Color", list(spore_color_display.keys()))

# Predict when button is clicked
if st.button("🔍 Predict"):
    input_row = {
        'cap-shape': cap_shape_display[cap_shape],
        'odor': odor_display[odor],
        'gill-size': gill_size_display[gill_size],
        'spore-print-color': spore_color_display[spore_color]
    }

    # Encode input using same encoders as training
    input_encoded = [label_encoders[col].transform([input_row[col]])[0] for col in features]

    prediction = model.predict([input_encoded])[0]

    if prediction == 0:
        st.success("✅ This mushroom is **edible**.")
    else:
        st.error("☠️ This mushroom is **poisonous**.")
