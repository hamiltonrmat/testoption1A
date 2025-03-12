import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Charger le modèle
model = load_model("keras_Model.h5", compile=False)

# Charger les étiquettes des classes
class_names = [line.strip() for line in open("labels.txt", "r").readlines()]

def preprocess_image(image):
    """Prétraitement de l'image pour le modèle."""
    image = image.resize((224, 224))
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1  # Normalisation
    return image

def predict(image):
    """Prédire la classe de l'image."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# Interface utilisateur Streamlit
st.title("Classification des Maladies des Plantes")
st.write("Chargez une image pour détecter d'éventuelles maladies.")

# Upload de fichier
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

# Capture d'image depuis la caméra
use_camera = st.checkbox("Utiliser la caméra")
if use_camera:
    picture = st.camera_input("Prenez une photo")
    if picture:
        uploaded_file = picture

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée", use_column_width=True)
    
    # Prédiction
    class_name, confidence_score = predict(image)
    
    # Affichage des résultats
    st.write(f"**Classe prédite :** {class_name}")
    st.write(f"**Score de confiance :** {confidence_score * 100:.2f}%")
