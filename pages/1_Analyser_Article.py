# ============================================================
# PAGE 1 : ANALYSER UN ARTICLE
# ============================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys
import os

# --- GESTION DE L'IMPORT DE UTILS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import init_page

# ============================================================
# CONFIGURATION DE LA PAGE ET CHARGEMENT CSS
# ============================================================

init_page(page_title="Analyse d'Article")

# ============================================================
# CHARGEMENT DES MODÈLES
# ============================================================

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
    st.error("❌ Impossible de charger les modèles depuis le dossier 'models/'.")

vader = SentimentIntensityAnalyzer()

# ============================================================
# FONCTIONS D’ANALYSE
# ============================================================

def analyser_sentiment(texte):
    scores = vader.polarity_scores(texte)
    compound = scores["compound"]
    if compound >= 0.05:
        label, emoji = "Positif", "😊"
    elif compound <= -0.05:
        label, emoji = "Négatif", "😠"
    else:
        label, emoji = "Neutre", "😐"
    return {"score": compound, "label": label, "emoji": emoji, "details": scores}

def analyser_subjectivite(texte):
    score = TextBlob(texte).sentiment.subjectivity
    if score >= 0.6:
        label, emoji = "Très subjectif", "📝"
    elif score >= 0.4:
        label, emoji = "Modérément subjectif", "📋"
    else:
        label, emoji = "Objectif", "📰"
    return {"score": score, "label": label, "emoji": emoji}

def analyser_toxicite(texte, sentiment_score):
    mots_toxiques = ["hate", "stupid", "idiot", "kill", "fake", "hoax",
                     "liar", "fraud", "scam", "corrupt"]
    texte_lower = texte.lower()
    mots_detectes = [m for m in mots_toxiques if m in texte_lower]
    score = min((len(mots_detectes) * 0.1) + abs(min(sentiment_score, 0)), 1)
    if score >= 0.5:
        label, emoji, is_toxic = "Toxique", "☠️", True
    elif score >= 0.25:
        label, emoji, is_toxic = "Légèrement toxique", "⚠️", True
    else:
        label, emoji, is_toxic = "Non toxique", "✅", False
    return {"is_toxic": is_toxic, "score": score, "label": label, "emoji": emoji, "mots_detectes": mots_detectes}

def analyser_bot(texte):
    mots = texte.lower().split()
    if len(mots) < 5:
        return {"is_bot": False, "score": 0, "label": "Texte trop court", "emoji": "❓", "raisons": []}
    diversite = len(set(mots)) / len(mots)
    score = 0
    raisons = []
    if diversite < 0.4:
        score += 0.4
        raisons.append("Vocabulaire très répétitif")
    if len(mots) < 20:
        score += 0.2
        raisons.append("Texte très court")
    if score >= 0.5:
        label, emoji, is_bot = "Probablement un bot", "🤖", True
    elif score >= 0.3:
        label, emoji, is_bot = "Comportement suspect", "🤔", False
    else:
        label, emoji, is_bot = "Comportement humain", "👤", False
    return {"is_bot": is_bot, "score": min(score,1), "label": label, "emoji": emoji, "raisons": raisons}

def predire_fake_news(texte, features):
    X_text = tfidf.transform([texte])
    X_num = np.array([[features["sentiment"], features["subjectivite"], features["toxicite"], features["bot"], features["word_count"], 5, 0.5]])
    X_num_scaled = scaler.transform(X_num)
    X_final = hstack([X_text, csr_matrix(X_num_scaled)])
    pred = model.predict(X_final)[0]
    proba = model.predict_proba(X_final)[0]
    return {"prediction": "Real News" if pred==1 else "Fake News",
            "is_fake": pred==0,
            "proba_fake": proba[0],
            "proba_real": proba[1],
            "confiance": max(proba)}

def analyse_complete(texte):
    sentiment = analyser_sentiment(texte)
    subjectivite = analyser_subjectivite(texte)
    toxicite = analyser_toxicite(texte, sentiment["score"])
    bot = analyser_bot(texte)
    features = {
        "sentiment": sentiment["score"],
        "subjectivite": subjectivite["score"],
        "toxicite": int(toxicite["is_toxic"]),
        "bot": int(bot["is_bot"]),
        "word_count": len(texte.split())
    }
    prediction = predire_fake_news(texte, features)
    return {"sentiment": sentiment, "subjectivite": subjectivite, "toxicite": toxicite, "bot": bot, "prediction": prediction, "word_count": features["word_count"]}

# ============================================================
# INTERFACE UTILISATEUR
# ============================================================

st.markdown('<p class="main-header">🔍 Analyseur d\'Intégrité</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Évaluation sémantique et prédiction de fiabilité par Intelligence Artificielle</p>', unsafe_allow_html=True)

# Zone de saisie stylisée
st.markdown("### 📝 Contenu à vérifier")
texte_input = st.text_area("", height=250, placeholder="Collez ici l'article (en anglais)...", label_visibility="collapsed")

# Bouton centré
col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
with col_c2:
    bouton_analyse = st.button("Lancer l'analyse", type="primary", use_container_width=True)

if bouton_analyse:

    if not MODELES_CHARGES:
        st.error("Les modèles ne sont pas chargés.")
    elif texte_input.strip() == "":
        st.warning("Veuillez saisir un texte.")
    else:
        with st.spinner("Analyse en cours..."):
            resultats = analyse_complete(texte_input)

        prediction = resultats["prediction"]

        if prediction["is_fake"]:
            st.error(f"⚠️ **FAKE NEWS** — Confiance : {prediction['confiance']:.1%}")
        else:
            st.success(f"✅ **ARTICLE AUTHENTIQUE** — Confiance : {prediction['confiance']:.1%}")

        recap = pd.DataFrame({
            "Analyse": ["Authenticité", "Sentiment", "Subjectivité", "Toxicité", "Bot"],
            "Résultat": [
                prediction["prediction"],
                resultats["sentiment"]["label"],
                resultats["subjectivite"]["label"],
                resultats["toxicite"]["label"],
                resultats["bot"]["label"]
            ],
            "Score": [
                f"{prediction['confiance']:.1%}",
                f"{resultats['sentiment']['score']:.2f}",
                f"{resultats['subjectivite']['score']:.2f}",
                f"{resultats['toxicite']['score']:.2f}",
                f"{resultats['bot']['score']:.2f}"
            ]
        })

        st.dataframe(recap, use_container_width=True, hide_index=True)
