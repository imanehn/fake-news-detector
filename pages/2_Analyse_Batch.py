# ============================================================
# PAGE 2 : ANALYSE PAR LOT (BATCH)
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys
import os

# --- GESTION DE L'IMPORT DE UTILS ---
# Cette partie permet de trouver utils.py même si on est dans le dossier /pages
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import de la configuration centralisée
from utils import init_page

# ------------------------------------------------------------
# CONFIGURATION ET CHARGEMENT
# ------------------------------------------------------------

init_page(page_title="Analyse Batch - Fake News Detector")

@st.cache_resource
def charger_modeles():
    model = joblib.load("models/xgb_model.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, tfidf, scaler

try:
    model, tfidf, scaler = charger_modeles()
    MODELES_CHARGES = True
except Exception:
    MODELES_CHARGES = False
    st.error("❌ Impossible de charger les modèles.")

vader = SentimentIntensityAnalyzer()

# ------------------------------------------------------------
# FONCTION D'ANALYSE (MODE BATCH)
# ------------------------------------------------------------

def analyser_article_batch(texte):
    mots = texte.lower().split()
    sentiment = vader.polarity_scores(texte)["compound"]
    subjectivite = TextBlob(texte).sentiment.subjectivity
    is_toxic = int(sentiment < -0.5)

    if len(mots) > 5:
        diversite = len(set(mots)) / len(mots)
        is_bot = int(diversite < 0.45)
    else:
        is_bot = 0

    # Préparation des features (doit correspondre à l'entraînement du modèle)
    X_text = tfidf.transform([texte])
    X_num = np.array([[sentiment, subjectivite, is_toxic, is_bot, len(mots), 5, 0.5]])
    X_num_scaled = scaler.transform(X_num)
    X_final = hstack([X_text, csr_matrix(X_num_scaled)])

    prediction = model.predict(X_final)[0]
    proba = model.predict_proba(X_final)[0]

    return {
        "prediction": "Real" if prediction == 1 else "Fake",
        "confiance": max(proba),
        "sentiment": sentiment,
        "subjectivite": subjectivite,
        "toxicite": "Oui" if is_toxic else "Non",
        "bot": "Oui" if is_bot else "Non"
    }

# ------------------------------------------------------------
# INTERFACE UTILISATEUR
# ------------------------------------------------------------

st.markdown('<p class="main-header">📊 Analyse par Lot</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Traitez des volumes importants de données instantanément</p>', unsafe_allow_html=True)

# Zone de téléversement
st.markdown("### 📥 Importer les données")
uploaded_file = st.file_uploader("Glissez-déposez votre fichier CSV ici", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Aperçu
        with st.expander("👁️ Aperçu du fichier source", expanded=False):
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        
        text_column = st.selectbox(
            "Quelle colonne contient les articles à analyser ?",
            df.columns
        )
        
        # Bouton centré
        col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
        with col_c2:
            btn_run = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)

        if btn_run:
            if not MODELES_CHARGES:
                st.error("Les modèles ne sont pas chargés.")
            else:
                with st.spinner("Analyse en cours..."):
                    resultats = []
                    total = len(df)

                    # Boucle de traitement
                    for i, texte in enumerate(df[text_column].astype(str)):
                        if len(texte.strip()) > 10:
                            res = analyser_article_batch(texte)
                            res["Aperçu"] = texte[:60] + "..."
                        else:
                            res = {"Aperçu": "Texte invalide", "prediction": "Erreur", "confiance": 0, 
                                   "sentiment": 0, "subjectivite": 0, "toxicite": "N/A", "bot": "N/A"}
                        
                        resultats.append(res)

                df_res = pd.DataFrame(resultats)
                
                # Statistiques simples
                fakes = len(df_res[df_res["prediction"]=="Fake"])
                reals = len(df_res[df_res["prediction"]=="Real"])
                
                if fakes > reals:
                    st.error(f"⚠️ **{fakes} Fake News détectées** sur {total} articles analysés")
                else:
                    st.success(f"✅ **{reals} articles authentiques** sur {total} articles analysés")

                # DataFrame des résultats
                st.dataframe(df_res, use_container_width=True, hide_index=True)
                
                # Export
                csv = df_res.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Exporter les résultats en CSV",
                    data=csv,
                    file_name="analyse_batch_results.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement : {e}")