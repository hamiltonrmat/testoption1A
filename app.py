import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import os
import sys

sys.setrecursionlimit(5000)

import tensorflow as tf



import os

# Obtenir le chemin du répertoire actuel
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construire le chemin complet vers le modèle
model_path = os.path.join("keras_Model.h5")
labels_path = os.path.join("labels.txt")

# Charger le modèle
model = load_model(model_path, compile=False, safe_mode=False)



from tensorflow.keras.models import load_model


import tensorflow as tf
from tensorflow.keras.models import load_model

def load_model_with_custom_objects(model_path):
    # Définir les objets personnalisés pour gérer les incompatibilités
    custom_objects = {}
    
    # Patch pour DepthwiseConv2D
    if hasattr(tf.keras.layers, 'DepthwiseConv2D'):
        original_init = tf.keras.layers.DepthwiseConv2D.__init__
        
        def patched_init(self, *args, **kwargs):
            # Supprimer le paramètre 'groups' s'il existe
            if 'groups' in kwargs:
                del kwargs['groups']
            return original_init(self, *args, **kwargs)
        
        tf.keras.layers.DepthwiseConv2D.__init__ = patched_init
    
    # Charger le modèle
    model = load_model(model_path, compile=False, custom_objects=custom_objects)
    return model

# Utiliser la fonction personnalisée
model = load_model_with_custom_objects("keras_model.h5")

# Configuration de la page
st.set_page_config(page_title="Classificateur d'Images", layout="wide")

# Titre de l'application
st.title("Classificateur d'Images")

# Description
st.markdown("""
Cette application utilise un modèle préentraîné pour classifier des images.
Téléchargez une image et l'application prédira sa classe avec un score de confiance.
""")

model = load_model("keras_Model.h5", compile=False, safe_mode=False)

# Fonction pour charger et classifier une image
def classify_image(image, model_path="keras_Model.h5", labels_path="labels.txt"):
    # Préparer l'image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    # Normaliser l'image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Créer un batch avec une seule image
    data = np.expand_dims(normalized_image_array, axis=0)
    
    # Charger les labels
    class_names = open(labels_path, "r").readlines()
    
    # Utiliser TensorFlow de manière directe pour éviter les problèmes de récursion
    try:
        # Créer une session TensorFlow et charger le modèle
        # Approche avec tf.keras simplifiée
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Faire la prédiction
        prediction = model.predict(data)
        
        # Obtenir la classe avec la plus haute probabilité
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        
        return class_name[2:], confidence_score
    except Exception as e:
        st.error(f"Erreur lors de la classification: {e}")
        # Essayer une approche alternative si la première échoue
        try:
            st.warning("Tentative avec une approche alternative...")
            # Approche alternative - charger le modèle en tant que SavedModel
            import tensorflow.compat.v1 as tf1
            tf1.disable_eager_execution()
            
            # Créer un répertoire temporaire pour le SavedModel
            saved_model_dir = os.path.splitext(model_path)[0] + "_saved_model"
            
            # Vérifier si le SavedModel existe déjà
            if not os.path.exists(saved_model_dir):
                st.info("Conversion du modèle en SavedModel...")
                # Convertir le modèle h5 en SavedModel
                model = tf.keras.models.load_model(model_path, compile=False)
                tf.saved_model.save(model, saved_model_dir)
            
            # Charger le SavedModel
            with tf1.Session() as sess:
                tf1.saved_model.loader.load(sess, ["serve"], saved_model_dir)
                
                # Obtenir les noms des tenseurs d'entrée et de sortie
                input_tensor = sess.graph.get_tensor_by_name("serving_default_input_1:0")
                output_tensor = sess.graph.get_tensor_by_name("StatefulPartitionedCall:0")
                
                # Faire la prédiction
                prediction = sess.run(output_tensor, {input_tensor: data})
                
                # Obtenir la classe avec la plus haute probabilité
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]
                
                return class_name[2:], confidence_score
        except Exception as e2:
            st.error(f"Erreur lors de l'approche alternative: {e2}")
            raise Exception(f"Impossible de classifier l'image avec le modèle fourni. Erreur initiale: {e}. Erreur alternative: {e2}")

# Interface utilisateur pour télécharger une image
uploaded_file = st.file_uploader("Choisissez une image à classifier", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image téléchargée", use_container_width=True)
    
    # Configuration du modèle
    model_path = st.sidebar.text_input("Chemin du modèle", "keras_Model.h5")
    labels_path = st.sidebar.text_input("Chemin des étiquettes", "labels.txt")
    
    # Vérifier l'existence des fichiers
    model_exists = os.path.exists(model_path)
    labels_exist = os.path.exists(labels_path)
    
    if not model_exists:
        st.sidebar.error(f"Le fichier modèle '{model_path}' n'existe pas")
    if not labels_exist:
        st.sidebar.error(f"Le fichier des étiquettes '{labels_path}' n'existe pas")
    
    # Ajouter un bouton pour lancer la classification
    if st.button("Classifier l'image"):
        if model_exists and labels_exist:
            with st.spinner("Classification en cours..."):
                try:
                    # Faire la prédiction
                    class_name, confidence_score = classify_image(image, model_path, labels_path)
                    
                    # Afficher les résultats
                    st.success("Classification terminée!")
                    
                    # Créer deux colonnes pour afficher les résultats
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(label="Classe prédite", value=class_name)
                    
                    with col2:
                        st.metric(label="Score de confiance", value=f"{confidence_score:.4f}")
                    
                    # Afficher une barre de progression pour visualiser le score de confiance
                    st.progress(float(confidence_score))
                    
                    # Afficher des informations supplémentaires sur la prédiction
                    st.subheader("Détails de la prédiction")
                    st.write(f"Le modèle a identifié cette image comme appartenant à la classe **{class_name}** avec un score de confiance de **{confidence_score:.4f}** (ou **{confidence_score*100:.2f}%**).")
                except Exception as e:
                    st.error(f"Erreur lors de la classification: {e}")
        else:
            st.error("Veuillez vérifier les chemins des fichiers modèle et étiquettes.")
else:
    st.info("Veuillez télécharger une image pour commencer la classification.")

# Ajouter de l'information sur le modèle utilisé
st.sidebar.header("Information sur le modèle")
st.sidebar.info("""
Ce classificateur utilise un modèle Keras/TensorFlow.
- Format d'entrée: images RGB 224x224
- Normalisation: (pixel / 127.5) - 1
""")

# Ajouter des instructions supplémentaires
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Téléchargez une image en utilisant le sélecteur de fichiers
2. Vérifiez que l'image s'affiche correctement
3. Cliquez sur "Classifier l'image"
4. Examinez les résultats de la classification
""")

# Afficher les informations techniques
st.sidebar.header("Informations techniques")
st.sidebar.text(f"TensorFlow version: {tf.__version__}")
st.sidebar.text(f"Python version: {sys.version.split()[0]}")

# Afficher les options avancées
st.sidebar.header("Options avancées")
if st.sidebar.checkbox("Afficher les options avancées"):
    st.sidebar.warning("Modification de ces options peut affecter la performance du modèle.")
    # Ajouter ici des options avancées si nécessaire
