import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import numpy as np

# ══════════════════════════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Système de Prédiction du Turnover",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# CSS PERSONNALISÉ
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }

    .risk-high {
        background-color: #fff5f5;
        border-left-color: #dc3545;
        color: #2d2d2d;
    }

    .risk-high h2 {
        color: #dc3545;
        font-weight: bold;
    }

    .risk-high h3 {
        color: #721c24;
        font-weight: bold;
    }

    .risk-high p,
    .risk-high ul,
    .risk-high li {
        color: #2d2d2d;
        font-size: 1rem;
    }

    .risk-low {
        background-color: #f0fff4;
        border-left-color: #28a745;
        color: #2d2d2d;
    }

    .risk-low h2 {
        color: #28a745;
        font-weight: bold;
    }

    .risk-low h3 {
        color: #155724;
        font-weight: bold;
    }

    .risk-low p,
    .risk-low ul,
    .risk-low li {
        color: #2d2d2d;
        font-size: 1rem;
    }

    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #155a8a;
    }

    div.stButton > button {
        height: 80px;
        font-size: 22px;
        font-weight: bold;
        padding: 30px !important;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CHARGER LE MODÈLE, CONFIG ET ENCODEURS
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("config.pkl", "rb") as f:
            config = pickle.load(f)

        with open("label_encoders.pkl", "rb") as f:
            encoders = pickle.load(f)

        return model, config, encoders

    except FileNotFoundError:
        st.warning(
            "⚠️ Fichiers du modèle introuvables ! "
            "Veuillez ajouter model.pkl, config.pkl et label_encoders.pkl."
        )
        return None, None, None


model, config, encoders = load_model()

if encoders:
    direction_names = encoders["le_direction"].classes_.tolist()
    subdept_names = encoders["le_subdept"].classes_.tolist()
else:
    direction_names = ["Direction A", "Direction B", "Direction C"]
    subdept_names = ["Dept 1", "Dept 2", "Dept 3"]

# ══════════════════════════════════════════════════════════════
# EN-TÊTE
# ══════════════════════════════════════════════════════════════

st.markdown(
    '<h1 class="main-header">🎯 Système de Prédiction du Turnover des Employés</h1>',
    unsafe_allow_html=True
)

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📝 Prédiction",
    "📊 Performance du Modèle",
    "ℹ️ À propos"
])

# ══════════════════════════════════════════════════════════════
# ONGLET 1 : FORMULAIRE DE PRÉDICTION
# ══════════════════════════════════════════════════════════════

with tab1:
    st.header("Formulaire d'Information sur l'Employé")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Performance & Évaluation")

        epns = st.selectbox(
            "EPNS Rate 2025",
            options=["Promoters", "Passives", "Detractors"],
            help="Score de satisfaction et d'engagement de l'employé."
        )

        perf_2025 = st.selectbox(
            "Performance rate 2025",
            options=[
                "Outstanding",
                "Exceed Expectations",
                "Solid Performance",
                "Marginally below expectations",
                "No Performance"
            ],
            help="Notation de performance pour 2025."
        )

        perf_2024 = st.selectbox(
            "Performance rate 2024",
            options=[
                "Outstanding",
                "Exceed Expectations",
                "Solid Performance",
                "Marginally below expectations",
                "No Performance"
            ],
            help="Notation de performance pour 2024."
        )

        perf_2023 = st.selectbox(
            "Performance rate 2023",
            options=[
                "Outstanding",
                "Exceed Expectations",
                "Solid Performance",
                "Marginally below expectations",
                "No Performance"
            ],
            help="Notation de performance pour 2023."
        )

        promotion = st.selectbox(
            "Promotion Last 2 Years",
            options=["No", "Yes"],
            help="Indique si l'employé a été promu au cours des 2 dernières années."
        )

        salary = st.selectbox(
            "Salary Placement 2025",
            options=[
                "Marginally Below Placement",
                "Within the company Placement",
                "High"
            ],
            help="Position du salaire par rapport aux normes du marché."
        )

    with col2:
        st.subheader("👤 Informations Personnelles")

        age = st.selectbox(
            "Age Range",
            options=["20 & 25", "26 & 30", "31 & 40", "40 & 50", "50+"],
            help="Tranche d'âge de l'employé."
        )

        gender_display = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            help="Genre de l'employé."
        )

        manager = st.selectbox(
            "M/NM",
            options=["Manager", "No Manager"],
            help="Indique si l'employé occupe une position managériale."
        )

        education = st.selectbox(
            "Education level",
            options=["BAC", "Bachelor", "Engineer", "Master", "PhD"],
            help="Plus haut niveau d'études complété."
        )

        seniority = st.slider(
            "Seniority",
            min_value=0.0,
            max_value=40.0,
            value=0.0,
            step=0.5,
            help="Nombre d'années d'ancienneté dans l'entreprise."
        )

        direction = st.selectbox(
            "Direction",
            options=direction_names,
            help="Direction de l'employé."
        )

        subdept = st.selectbox(
            "Sub Dept",
            options=subdept_names,
            help="Sous-département de l'employé."
        )

    with col3:
        st.subheader("⚙️ Conditions de Travail")

        training_hours = st.slider(
            "Traning Hours",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            help="Nombre total d'heures de formation reçues."
        )

        stress = st.selectbox(
            "Stress Level",
            options=["Low Level", "Acceptable Level", "High Level"],
            help="Niveau de stress déclaré par l'employé."
        )

        overload = st.selectbox(
            "Work overload",
            options=["No", "Yes"],
            help="Indique si l'employé subit une surcharge de travail."
        )

        transport = st.selectbox(
            "Transportation Way",
            options=["Company Transportation", "Private Transportation"],
            help="Mode de transport utilisé par l'employé."
        )

        scheme = st.selectbox(
            "Working Scheme",
            options=["FT", "PT", "TH"],
            help="Schéma de travail de l'employé."
        )

    st.markdown("---")

    col_btn1, col_btn2, col_btn3 = st.columns([0.5, 2, 0.5])

    with col_btn2:
        predict_clicked = st.button(
            "🎯 PRÉDIRE LE RISQUE DE DÉPART",
            type="primary",
            use_container_width=True
        )

    if predict_clicked:

        if model is None or config is None:
            st.error("Modèle non chargé. Veuillez vérifier les fichiers du modèle.")

        else:
            mappings = config["mappings"]

            if encoders:
                direction_encoded = encoders["le_direction"].transform([direction])[0]
                subdept_encoded = encoders["le_subdept"].transform([subdept])[0]
            else:
                direction_encoded = 0
                subdept_encoded = 0

            gender = "H" if gender_display == "Male" else "F"

            input_data = pd.DataFrame({
                "Direction": [direction_encoded],
                "Sub Dept": [subdept_encoded],
                "EPNS Rate 2025": [mappings["epns"][epns]],
                "Performance rate 2025": [mappings["performance"][perf_2025]],
                "Performance rate 2024": [mappings["performance"][perf_2024]],
                "Performance rate 2023": [mappings["performance"][perf_2023]],
                "Promotion Last 2 Years": [mappings["binary"][promotion]],
                "Salary Placement 2025": [mappings["salary"][salary]],
                "Age Range": [mappings["age"][age]],
                "Gender": [mappings["gender"][gender]],
                "M/NM": [mappings["manager"][manager]],
                "Traning Hours": [training_hours],
                "Stress Level": [mappings["stress"][stress]],
                "Seniority": [seniority],
                "Work overload": [mappings["binary"][overload]],
                "Education level": [mappings["education"][education]],
                "Transportation Way": [mappings["transport"][transport]],
                "Working Scheme": [mappings["scheme"][scheme]]
            })

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            prob_stay = probability[0] * 100
            prob_leave = probability[1] * 100

            st.markdown("---")
            st.header("📊 Résultats de la Prédiction")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Risque de Départ",
                    f"{prob_leave:.1f}%",
                    delta=f"{prob_leave - 50:.1f}% vs référence",
                    delta_color="inverse"
                )
                st.progress(prob_leave / 100)

            with result_col2:
                st.metric(
                    "Probabilité de Rétention",
                    f"{prob_stay:.1f}%",
                    delta=f"{prob_stay - 50:.1f}% vs référence"
                )
                st.progress(prob_stay / 100)

            chart_data = pd.DataFrame({
                "Catégorie": ["Restera", "Partira"],
                "Probabilité": [prob_stay, prob_leave]
            })

            st.bar_chart(chart_data.set_index("Catégorie"))

            st.markdown("---")

            if prediction == 1:
                st.markdown(f"""
                <div class="metric-card risk-high">
                    <h2>⚠️ RISQUE ÉLEVÉ - L'employé est susceptible de partir</h2>
                    <h3>Score de Risque : {prob_leave:.1f}%</h3>
                    <p><strong>Actions Recommandées :</strong></p>
                    <ul>
                        <li><strong>🎯 Intervention RH immédiate requise</strong></li>
                        <li><strong>💬 Programmer un entretien individuel</strong></li>
                        <li><strong>📊 Examiner Work overload et Stress Level</strong></li>
                        <li><strong>🎓 Envisager des opportunités de formation</strong></li>
                        <li><strong>💰 Évaluer Salary Placement et avantages</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="metric-card risk-low">
                    <h2>✅ RISQUE FAIBLE - L'employé est susceptible de rester</h2>
                    <h3>Score de Rétention : {prob_stay:.1f}%</h3>
                    <p><strong>Actions Recommandées :</strong></p>
                    <ul>
                        <li><strong>✅ Continuer le suivi régulier</strong></li>
                        <li><strong>🎯 Maintenir les stratégies d'engagement</strong></li>
                        <li><strong>📈 Encourager le développement de carrière</strong></li>
                        <li><strong>🏆 Reconnaître les bonnes performances</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ONGLET 2 : PERFORMANCE DU MODÈLE
# ══════════════════════════════════════════════════════════════

with tab2:
    st.header("📊 Métriques de Performance du Modèle")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Accuracy", "97.30%", "↑ Excellent")

    with col2:
        st.metric("AUC-ROC", "99.62%", "↑ Quasi-Parfait")

    with col3:
        st.metric("Sensitivity", "93.24%", "↑ Très élevée")

    with col4:
        st.metric("Specificity", "98.46%", "↑ Excellente")

    with col5:
        st.metric("F1-Score", "93.88%", "↑ Excellent")

    st.markdown("---")

    st.subheader("🔍 Principales Features influençant le Turnover")

    features_data = pd.DataFrame({
        "Feature": [
            "Stress Level",
            "Traning Hours",
            "Performance rate 2025",
            "Work overload",
            "Performance rate 2023",
            "EPNS Rate 2025",
            "Promotion Last 2 Years"
        ],
        "Importance": [
            12.3,
            12.2,
            11.8,
            10.7,
            9.9,
            9.7,
            6.7
        ]
    })

    features_data = features_data.sort_values("Importance", ascending=True)

    st.bar_chart(features_data.set_index("Feature"))

    st.info("""
    **Insights Clés :**

    - **Stress Level** fait partie des facteurs les plus influents dans le risque de départ.
    - **Traning Hours** impacte fortement la rétention des employés.
    - **Work overload** augmente significativement le risque de turnover.
    - **Performance rate 2025**, **Performance rate 2023** et **EPNS Rate 2025** jouent également un rôle important.
    """)

# ══════════════════════════════════════════════════════════════
# ONGLET 3 : À PROPOS
# ══════════════════════════════════════════════════════════════

with tab3:
    st.header("ℹ️ À propos de ce Système")

    st.markdown(f"""
    ### 🎯 Objectif
    Système de Prédiction du Turnover des Employés utilisant le **Machine Learning**
    pour identifier les employés à risque et permettre une intervention RH proactive.

    ### 🤖 Détails du Modèle
    - **Algorithme :** Random Forest Classifier
    - **Données analysées :** 1,661 enregistrements d'employés
    - **Features utilisées :** 18 caractéristiques d'employés
    - **Performance :** 99.62% AUC-ROC

    ### 📊 Métriques Clés
    - **Accuracy :** 97.30%
    - **AUC-ROC :** 99.62%
    - **Sensitivity :** 93.24%
    - **Specificity :** 98.46%
    - **Precision :** 94.52%
    - **F1-Score :** 93.88%

    ### 📋 Matrice de Confusion
    - **TN :** 255
    - **FP :** 4
    - **FN :** 5
    - **TP :** 69

    ### 🔍 Principales Features de Départ
    1. Stress Level (12.3%)
    2. Traning Hours (12.2%)
    3. Performance rate 2025 (11.8%)
    4. Work overload (10.7%)
    5. Performance rate 2023 (9.9%)
    6. EPNS Rate 2025 (9.7%)
    7. Promotion Last 2 Years (6.7%)

    ### 📈 Version
    **v1.0** - {datetime.now().strftime("%B %Y")}

    ---

    *Développé avec Python, Scikit-learn & Streamlit*
    """)

# ══════════════════════════════════════════════════════════════
# BARRE LATÉRALE
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📊 Prédicteur de Turnover")

    st.markdown("---")

    st.subheader("📊 Statistiques Rapides")

    st.metric("Modèle", "Random Forest")
    st.metric("Précision du Modèle", "97.30%")
    st.metric("AUC-ROC", "99.62%")
    st.metric("Employés Analysés", "1,661")
    st.metric("Taux de Turnover", "22.22%")

    st.markdown("---")

    st.caption(f"Dernière mise à jour : {datetime.now().strftime('%Y-%m-%d')}")
    st.caption("Alimenté par Random Forest ML") 
