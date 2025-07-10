import streamlit as st
import pickle

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# image
st.image("mushroom.jpeg", use_container_width=True)

# Mapping input
cap_shape_display = {
    "Bell": 'b', "Conical": 'c', "Convex": 'x',
    "Flat": 'f', "Knobbed": 'k', "Sunken": 's'
}
odor_display = {
    "Almond": 'a', "Anise": 'l', "Creosote": 'c', "Fishy": 'y',
    "Foul": 'f', "Musty": 'm', "None": 'n', "Pungent": 'p', "Spicy": 's'
}
gill_size_display = {
    "Broad": 'b', "Narrow": 'n'
}
spore_color_display = {
    "Black": 'k', "Brown": 'n', "Buff": 'b', "Chocolate": 'h',
    "Green": 'r', "Orange": 'o', "Purple": 'u', "White": 'w', "Yellow": 'y'
}

# Label-encoding mappings
cap_map = {'b': 0, 'c': 1, 's': 2, 'f': 3, 'k': 4, 'x': 5}
odor_map = {'a': 0, 'c': 1, 'f': 2, 's': 3, 'y': 4, 'p': 5, 'l': 6, 'm': 7, 'n': 8}
gill_map = {'b': 0, 'n': 1}
spore_map = {'b': 0, 'h': 1, 'k': 2, 'n': 3, 'o': 4, 'r': 5, 'u': 6, 'w': 7, 'y': 8}

# Input fields
st.markdown("Mushroom Features")

cap_shape_input = st.selectbox("🧢 Cap Shape", list(cap_shape_display.keys()))
odor_input = st.selectbox("👃 Odor", list(odor_display.keys()))
gill_size_input = st.selectbox("🍃 Gill Size", list(gill_size_display.keys()))
spore_color_input = st.selectbox("🎨 Spore Print Color", list(spore_color_display.keys()))

# Convert input to encoded letters
cap_shape = cap_shape_display[cap_shape_input]
odor = odor_display[odor_input]
gill_size = gill_size_display[gill_size_input]
spore_color = spore_color_display[spore_color_input]

# Predict button
if st.button("Check Mushroom"):
    input_data = [
        cap_map[cap_shape],
        odor_map[odor],
        gill_map[gill_size],
        spore_map[spore_color]
    ]

    prediction = model.predict([input_data])[0]

    if prediction == 0:
        st.success("✅ This mushroom is **edible** ")
    else:
        st.error("☠️ This mushroom is **poisonous** ")

