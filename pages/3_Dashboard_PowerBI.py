# ============================================================
# PAGE 3 : DASHBOARD POWER BI
# ============================================================

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
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

init_page(page_title="Dashboard")
# ============================================================
# CONTENU PRINCIPAL
# ============================================================

st.markdown('<p class="main-header">Dashboard Power BI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Visualisations interactives issues des données analysées</p>', unsafe_allow_html=True)

# ============================================================
# Option 1 : Dashboards Power BI dans des Tabs
# ============================================================

tabs_dash = st.tabs(["Dashboard 1", "Dashboard 2"])

POWERBI_URL_1 = "https://app.powerbi.com/reportEmbed?reportId=eb6c3314-7eca-4208-83d7-128ad6b546a5&autoAuth=true&ctid=f93d5f40-88c0-4650-b8f2-cc4ec3ef6a10"
POWERBI_URL_2 = "https://app.powerbi.com/reportEmbed?reportId=fa03b2f8-8028-488d-bdbb-6e2df3263749&autoAuth=true&ctid=f93d5f40-88c0-4650-b8f2-cc4ec3ef6a10"

with tabs_dash[0]:
    components.iframe(src=POWERBI_URL_1, width=1200, height=700)

with tabs_dash[1]:
    components.iframe(src=POWERBI_URL_2, width=1200, height=700)
