import streamlit as st
import os

def load_css(file_name):
    """
    Fonction pour charger le CSS global sur toutes les pages
    Gère les chemins relatifs depuis n'importe quel emplacement
    """
    # Obtenir le répertoire racine du projet
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, file_name)
    
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Fichier CSS non trouvé: {css_path}")

def init_page(page_title="Fake News Detector"):
    """
    Configuration standard pour toutes les pages
    """
    st.set_page_config(
        page_title=page_title,
        page_icon="🔍",
        layout="wide"
    )
    # Chargement du CSS immédiatement après la config
    load_css("style.css")