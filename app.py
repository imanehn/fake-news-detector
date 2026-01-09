# ============================================================
# app.py - PAGE D'ACCUEIL
# ============================================================

import streamlit as st
from utils import init_page
from scipy.sparse import hstack
import numpy as np
import joblib

# ------------------------------------------------------------
# CONFIGURATION DE LA PAGE ET CHARGEMENT CSS
# ------------------------------------------------------------
init_page(page_title="Accueil - Détecteur")

# ------------------------------------------------------------
# INITIALISATION SESSION
# ------------------------------------------------------------
if "historique_analyses" not in st.session_state:
    st.session_state.historique_analyses = []
# ------------------------------------------------------------
# CONTENU PRINCIPAL
# ------------------------------------------------------------
st.markdown('<p class="main-header">Détecteur de Fake News</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyse intelligente des articles</p>', unsafe_allow_html=True)

# ------------------------------------------------------------
# FONCTIONNALITÉS
# ------------------------------------------------------------
st.header("Fonctionnalités")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""<div class="feature-card"><h3>🔍 Détection</h3><p>Détection des news avec plus de 96% de précision</p></div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="feature-card"><h3>😊 Sentiment</h3><p>Analyse le ton émotionnel de l'article</p></div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="feature-card"><h3>⚠️ Toxicité</h3><p>Détecte les contenus haineux ou toxiques</p></div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""<div class="feature-card"><h3>🤖 Bot</h3><p>Identifie les contenus générés par des bots</p></div>""", unsafe_allow_html=True)

# ------------------------------------------------------------
# COMMENT UTILISER
# ------------------------------------------------------------
st.markdown("---")
st.header("Comment utiliser")
st.markdown("""
1. Accédez à la page **"Analyser un article"** depuis le menu latéral  
2. Collez le texte de l'article à vérifier  
3. Cliquez sur **"Analyser"** pour obtenir les résultats  
4. Consultez les détails : prédiction, sentiment, toxicité, subjectivité et détection de bot  

Vous pouvez également :
- Analyser plusieurs articles via **l'analyse Batch**
- Explorer le **Dashboard**
""")

# ------------------------------------------------------------
# CHARGEMENT DU MODELE XGBOOST, TF-IDF, SCALER
# ------------------------------------------------------------
try:
    xgb_model = joblib.load("models/xgb_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    scaler = joblib.load("models/scaler.pkl")
    MODELES_CHARGES = True
    available_features = ['sentiment_score', 'subjectivity_score', 'is_toxic',
                          'is_potential_bot', 'word_count_clean', 'title_word_count', 'reduction_ratio']
except Exception as e:
    MODELES_CHARGES = False
    st.error(f"⚠️ Erreur lors du chargement des modèles : {e}")
    xgb_model = None
    tfidf = None
    scaler = None
    available_features = []

# ------------------------------------------------------------
# FONCTION DE PREDICTION AVEC SEUIL INTEGRE
# ------------------------------------------------------------
def predict_news_streamlit(news_text, numeric_features_dict, model, tfidf, scaler, available_features, threshold=0.15):
    text_vect = tfidf.transform([news_text])
    numeric_values = np.array([[numeric_features_dict[feat] for feat in available_features]])
    numeric_scaled = scaler.transform(numeric_values)
    X_input = hstack([text_vect, numeric_scaled])
    
    proba_real = model.predict_proba(X_input)[0][1]
    label = 'real' if proba_real >= threshold else 'fake'
    
    return label, proba_real

# ------------------------------------------------------------
# APPEL A L'ACTION
# ------------------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Commencer l'analyse", type="primary", use_container_width=True):
        st.switch_page("pages/1_Analyser_Article.py")