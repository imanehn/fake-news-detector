# ============================================================
# PAGE 5 : À PROPOS
# ============================================================

import streamlit as st
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

init_page(page_title="À propos")

# ============================================================
# CONTENU PRINCIPAL
# ============================================================

st.markdown('<p class="main-header">À propos du projet</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Détection automatique des Fake News par Intelligence Artificielle</p>', unsafe_allow_html=True)

# ------------------------------------------------------------
# DESCRIPTION DU PROJET
# ------------------------------------------------------------

st.header("Description du projet")

st.markdown("""
Ce projet vise la **détection automatique des Fake News** à partir d’articles textuels.

L’application développée permet :
- La prédiction de l’authenticité des articles (*Fake / Real*)
- L’analyse du **sentiment** (positif, négatif, neutre)
- L’évaluation du **degré de subjectivité**
- La détection de **contenus toxiques**
- L’identification de **comportements automatisés (bots)**
""")
# ------------------------------------------------------------
# PERFORMANCE DU MODÈLE (VERSION RÉSUMÉE)
# ------------------------------------------------------------
st.markdown("---")
st.header("Performance du modèle")

st.markdown("""
Le modèle **XGBoost**, entraîné sur un corpus de **39 000 articles**, 
a démontré d’excellentes performances globales.

Les résultats obtenus montrent une **forte capacité de discrimination**
entre les articles fiables et les fake news, garantissant ainsi
la **robustesse et la fiabilité** du système proposé.
""")
