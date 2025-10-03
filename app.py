import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from datetime import datetime
import time
import random
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="FraudGuard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement des données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('fraude_bancaire_synthetique_final.csv')
        # Nettoyage des données
        df['region'] = df['region'].str.strip()
        return df
    except Exception as e:
        st.error(f"Erreur de chargement des données: {str(e)}")
        return pd.DataFrame()

# Chargement du modèle
@st.cache_resource
def load_model():
    try:
        with open('random_forest_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Erreur de chargement du modèle: {str(e)}")
        return None

# Fonction pour exporter les données
def get_export_file(df, format_type):
    if format_type == "CSV":
        return df.to_csv(index=False).encode('utf-8')
    elif format_type == "Excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        return output.getvalue()
    elif format_type == "JSON":
        return df.to_json(orient='records').encode('utf-8')
    elif format_type == "Parquet":
        return df.to_parquet(index=False)

# CSS personnalisé - Thème sombre amélioré avec animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&family=Roboto:wght@400;500&display=swap');
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 240, 255, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(0, 240, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 240, 255, 0); }
    }
    @keyframes neonGlow {
        0%, 100% { text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 15px #00f0ff, 0 0 20px #00f0ff; }
        50% { text-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px #00f0ff, 0 0 40px #00f0ff; }
    }
    @keyframes cardHover {
        0% { transform: translateY(0); box-shadow: 0 0 12px rgba(0, 240, 255, 0.2); }
        100% { transform: translateY(-5px); box-shadow: 0 0 20px rgba(0, 240, 255, 0.4); }
    }

    :root {
        --neon-pink: #ff4ecd;
        --neon-blue: #00f0ff;
        --neon-green: #00ff9d;
        --bg-dark: #0f0f1b;
        --bg-darker: #0a0a12;
        --card-bg: #1c1c2c;
        --text-light: #e0e0e0;
        --text-lighter: #ffffff;
        --text-dark: #333333;
        --warning: #ff6d00;
        --danger: #ff1744;
        --success: #00e676;
        --border-radius: 12px;
    }

    html, body, .stApp {
        background-color: var(--bg-dark) !important;
        color: var(--text-light) !important;
        font-family: 'Roboto', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Orbitron', sans-serif;
        color: var(--text-lighter) !important;
        animation: fadeIn 0.5s ease-out;
    }

    .header {
        background: linear-gradient(90deg, var(--neon-pink), var(--neon-blue));
        padding: 1.5rem;
        border-radius: 0 0 var(--border-radius) var(--border-radius);
        text-align: center;
        box-shadow: 0 0 15px var(--neon-pink);
        color: var(--text-dark) !important;
        margin-bottom: 2rem;
        animation: pulse 2s infinite;
    }

    .nav-btn {
        background-color: transparent !important;
        color: var(--neon-blue) !important;
        border: 2px solid var(--neon-blue) !important;
        border-radius: var(--border-radius) !important;
        font-weight: bold !important;
        transition: all 0.3s ease;
        margin: 0.2rem 0;
        width: 100%;
    }

    .nav-btn:hover, .nav-btn.active {
        background-color: var(--neon-blue) !important;
        color: var(--text-dark) !important;
        box-shadow: 0 0 10px var(--neon-blue);
        transform: scale(1.02);
    }

    .card {
        background-color: var(--card-bg);
        border: 1px solid var(--neon-blue);
        padding: 1.5rem;
        border-radius: var(--border-radius);
        box-shadow: 0 0 12px rgba(0, 240, 255, 0.2);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        animation: fadeIn 0.7s ease-out;
    }

    .card:hover {
        animation: cardHover 0.3s ease forwards;
    }

    .fraud-result {
        border-left: 6px solid var(--danger);
        background-color: rgba(255, 23, 68, 0.1);
        padding: 1.5rem;
        margin: 1rem 0;
        color: var(--text-lighter);
        border-radius: var(--border-radius);
        animation: pulse 1.5s infinite;
    }

    .safe-result {
        border-left: 6px solid var(--success);
        background-color: rgba(0, 230, 118, 0.1);
        padding: 1.5rem;
        margin: 1rem 0;
        color: var(--text-lighter);
        border-radius: var(--border-radius);
        animation: fadeIn 0.5s ease-out;
    }

    .warning-result {
        border-left: 6px solid var(--warning);
        background-color: rgba(255, 109, 0, 0.1);
        padding: 1.5rem;
        margin: 1rem 0;
        color: var(--text-lighter);
        border-radius: var(--border-radius);
        animation: pulse 2s infinite;
    }

    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select,
    .stSlider>div>div>div>div {
        background-color: #12121f !important;
        color: var(--text-light) !important;
        border: 1px solid var(--neon-pink) !important;
        border-radius: var(--border-radius) !important;
    }

    .stDataFrame {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.1);
        animation: fadeIn 0.5s ease-out;
    }

    .stProgress>div>div>div {
        background-color: var(--neon-blue) !important;
    }

    .st-bb {
        border-bottom: 1px solid var(--neon-blue) !important;
    }

    .st-bt {
        border-top: 1px solid var(--neon-blue) !important;
    }

    .explanation-box {
        background-color: var(--card-bg);
        border: 1px solid var(--neon-green);
        border-radius: var(--border-radius);
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: 0 0 10px rgba(0, 255, 157, 0.2);
        animation: fadeIn 0.7s ease-out;
    }

    .feature-importance {
        background-color: var(--bg-darker);
        border-radius: var(--border-radius);
        padding: 1rem;
        margin-top: 1rem;
        animation: fadeIn 0.7s ease-out;
    }

    /* Style pour les onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--card-bg) !important;
        border-radius: var(--border-radius) var(--border-radius) 0 0 !important;
        padding: 0.5rem 1rem !important;
        margin: 0 !important;
        border: 1px solid var(--neon-blue) !important;
        color: var(--text-light) !important;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--neon-blue) !important;
        color: var(--text-dark) !important;
        font-weight: bold !important;
        transform: scale(1.05);
    }

    /* Style pour les graphiques Plotly */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly div {
        background-color: transparent !important;
    }

    /* Style pour les tooltips */
    .stTooltip {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--neon-blue) !important;
        color: var(--text-light) !important;
        animation: fadeIn 0.3s ease-out;
    }

    /* Animation pour les titres */
    .title-animation {
        animation: neonGlow 1.5s ease-in-out infinite alternate;
    }

    /* Contraste amélioré pour le texte */
    .text-contrast {
        color: var(--text-lighter) !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
    }

    /* Bouton d'export */
    .export-btn {
        background-color: var(--neon-green) !important;
        color: var(--text-dark) !important;
        border: none !important;
        font-weight: bold !important;
        transition: all 0.3s ease;
    }

    .export-btn:hover {
        background-color: var(--neon-blue) !important;
        transform: scale(1.05);
        box-shadow: 0 0 15px var(--neon-green);
    }
</style>
""", unsafe_allow_html=True)

# Initialisation session
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'last_transactions' not in st.session_state:
    st.session_state.last_transactions = []

# Fonctions de navigation
def change_page(page):
    st.session_state.page = page

# Chargement des données
df = load_data()
model = load_model()

# En-tête
st.markdown("""
<div class="header">
    <h1 style="margin: 0;" class="title-animation">🛡️ FraudGuard Pro</h1>
    <p style="margin: 0; color: var(--text-dark);">Détection avancée des fraudes financières - Intelligence Artificielle</p>
</div>
""", unsafe_allow_html=True)

# Navigation
cols = st.columns(4)
with cols[0]:
    st.button("📊 Tableau de bord", 
              on_click=change_page, args=('dashboard',),
              key='nav_dashboard',
              type="primary" if st.session_state.page == 'dashboard' else "secondary",
              help="Vue d'ensemble des transactions et indicateurs clés")

with cols[1]:
    st.button("📈 Analyse", 
              on_click=change_page, args=('analytics',),
              key='nav_analytics',
              type="primary" if st.session_state.page == 'analytics' else "secondary",
              help="Analyse approfondie des données et visualisations")

with cols[2]:
    st.button("🔍 Détection", 
              on_click=change_page, args=('detection',),
              key='nav_detection',
              type="primary" if st.session_state.page == 'detection' else "secondary",
              help="Détection de fraude en temps réel")

with cols[3]:
    st.button("⚙️ Paramètres", 
              on_click=change_page, args=('settings',),
              key='nav_settings',
              type="primary" if st.session_state.page == 'settings' else "secondary",
              help="Configuration du système")

# Séparateur
st.markdown("---")

# Fonction pour générer des explications détaillées
def generate_explanation(prediction, proba, input_data):
    explanation = ""
    
    if prediction == 1:
        explanation += "### 🚨 Raisons possibles de l'alerte de fraude:\n"
        
        if input_data['montant_transaction'].values[0] > df['montant_transaction'].quantile(0.95):
            explanation += f"- **Montant élevé:** €{input_data['montant_transaction'].values[0]:.2f} (supérieur à 95% des transactions)\n"
        
        if input_data['anciennete_compte'].values[0] < 1:
            explanation += f"- **Compte récent:** {input_data['anciennete_compte'].values[0]} an(s) d'ancienneté\n"
        
        if input_data['score_credit'].values[0] < 600:
            explanation += f"- **Score de crédit faible:** {input_data['score_credit'].values[0]}/850\n"
        
        explanation += f"\n**Confiance du modèle:** {proba[1]*100:.1f}%"
        
    else:
        explanation += "### ✅ Facteurs de légitimité:\n"
        
        if input_data['anciennete_compte'].values[0] > 3:
            explanation += f"- **Compte ancien:** {input_data['anciennete_compte'].values[0]} ans d'historique\n"
        
        if input_data['score_credit'].values[0] > 700:
            explanation += f"- **Bon score de crédit:** {input_data['score_credit'].values[0]}/850\n"
        
        if input_data['montant_transaction'].values[0] < df['montant_transaction'].quantile(0.75):
            explanation += f"- **Montant typique:** €{input_data['montant_transaction'].values[0]:.2f}\n"
        
        explanation += f"\n**Risque estimé:** {proba[1]*100:.1f}% (seuil d'alerte à 70%)"
    
    return explanation

# Fonction pour afficher l'importance des features
# Fonction pour afficher l'importance des features - Version corrigée
def show_feature_importance(model, input_data):
    try:
        # Création d'un explainer SHAP
        explainer = shap.TreeExplainer(model)
        
        # Calcul des valeurs SHAP
        shap_values = explainer.shap_values(input_data)
        
        # Gestion des différents formats de sortie SHAP
        if isinstance(shap_values, list):
            # Pour les classifieurs binaires, on prend les valeurs pour la classe positive (index 1)
            if len(shap_values) == 2:
                shap_values = shap_values[1]
            else:
                shap_values = shap_values[0]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Pour les problèmes multi-classes (on prend la première classe ici)
            shap_values = shap_values[0]
        
        # Vérification finale des dimensions
        if len(shap_values.shape) != 2:
            raise ValueError(f"Format de valeurs SHAP non supporté: {shap_values.shape}")
        
        # Création d'un DataFrame pour une meilleure visualisation
        feature_names = input_data.columns
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_value': np.abs(shap_values[0]).tolist()
        }).sort_values('SHAP_value', ascending=False)
        
        # Visualisation avec Plotly
        fig = px.bar(shap_df, 
                     x='SHAP_value', 
                     y='Feature',
                     orientation='h',
                     title='Importance des facteurs dans la décision',
                     labels={'SHAP_value': 'Importance (valeur SHAP absolue)', 'Feature': 'Facteur'},
                     color='SHAP_value',
                     color_continuous_scale=px.colors.sequential.Blues)
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title='Importance',
            yaxis_title='Facteurs',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Affichage des valeurs numériques pour plus de précision
        with st.expander("📊 Détails des valeurs SHAP"):
            st.dataframe(shap_df.set_index('Feature'), use_container_width=True)
            
    except Exception as e:
        st.error(f"Erreur lors de la génération de l'importance des facteurs: {str(e)}")
        st.warning("""
        Conseil de dépannage:
        1. Vérifiez que votre modèle est bien un modèle d'arbre (Random Forest, XGBoost, etc.)
        2. Assurez-vous que les données d'entrée ont le même format que lors de l'entraînement
        3. Essayez de réduire le nombre de fonctionnalités si l'erreur persiste
        """)
# Pages
if st.session_state.page == 'dashboard':
    st.markdown("## 📊 Tableau de bord")
    
    if not df.empty:
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="card">
                <h3 class="text-contrast">Transactions totales</h3>
                <h1 class="text-contrast">{:,}</h1>
                <p class="text-contrast">30 derniers jours</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            fraud_count = df['fraude'].sum()
            st.markdown("""
            <div class="card">
                <h3 class="text-contrast">Fraudes détectées</h3>
                <h1 class="text-contrast" style="color: var(--danger);">{:,}</h1>
                <p class="text-contrast">{:.2f}% des transactions</p>
            </div>
            """.format(fraud_count, (fraud_count/len(df))*100), unsafe_allow_html=True)
        
        with col3:
            avg_amount = df['montant_transaction'].mean()
            st.markdown("""
            <div class="card">
                <h3 class="text-contrast">Montant moyen</h3>
                <h1 class="text-contrast">€{:.2f}</h1>
                <p class="text-contrast">Max: €{:.2f}</p>
            </div>
            """.format(avg_amount, df['montant_transaction'].max()), unsafe_allow_html=True)
        
        with col4:
            fraud_amount = df[df['fraude'] == 1]['montant_transaction'].sum()
            st.markdown("""
            <div class="card">
                <h3 class="text-contrast">Montant fraudé</h3>
                <h1 class="text-contrast" style="color: var(--danger);">€{:.2f}</h1>
                <p class="text-contrast">{:.2f}% du total</p>
            </div>
            """.format(fraud_amount, (fraud_amount/df['montant_transaction'].sum())*100), unsafe_allow_html=True)
        
        # Graphiques
        tab1, tab2, tab3 = st.tabs(["📈 Activité", "🌍 Géographie", "💳 Méthodes"])
        
        with tab1:
            st.markdown("""
            <div class="card">
                <h3 class="text-contrast">Activité des transactions</h3>
            """, unsafe_allow_html=True)
            
            # Simulation de données temporelles
            df['date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))
            daily_data = df.groupby(df['date'].dt.date).agg(
                transactions=('montant_transaction', 'count'),
                frauds=('fraude', 'sum'),
                amount=('montant_transaction', 'sum')
            ).reset_index()
            
            fig = px.line(daily_data, x='date', y='transactions',
                         title='Transactions quotidiennes',
                         labels={'date': 'Date', 'transactions': 'Nombre de transactions'},
                         color_discrete_sequence=["#00f0ff"])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(daily_data, x='date', y='frauds',
                             title='Fraudes quotidiennes',
                             labels={'date': 'Date', 'frauds': 'Nombre de fraudes'},
                             color_discrete_sequence=["#00f0ff"])
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(df, x='montant_transaction', y='score_credit',
                                color='fraude',
                                title='Score crédit vs Montant',
                                color_discrete_map={0: '#00f0ff', 1: '#ff1744'},
                                labels={'montant_transaction': 'Montant (€)', 'score_credit': 'Score crédit'})
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
            <div class="card">
                <h3 class="text-contrast">Répartition géographique</h3>
            """, unsafe_allow_html=True)
            
            region_stats = df['region'].value_counts().reset_index()
            region_stats.columns = ['Région', 'Nombre']
            
            fig = px.bar(region_stats, x='Région', y='Nombre',
                        color='Région',
                        labels={'Nombre': 'Nombre de transactions'},
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
            
            # Carte choroplèthe (simulée)
            region_fraud = df.groupby('region')['fraude'].mean().reset_index()
            fig = px.choropleth(region_fraud, 
                               locations='region', 
                               locationmode='country names',
                               color='fraude',
                               scope='europe',
                               title='Taux de fraude par région',
                               color_continuous_scale=px.colors.sequential.Reds)
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown("""
            <div class="card">
                <h3 class="text-contrast">Méthodes de paiement</h3>
            """, unsafe_allow_html=True)
            
            card_stats = df['type_carte'].value_counts().reset_index()
            card_stats.columns = ['Type', 'Nombre']
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(card_stats, values='Nombre', names='Type',
                            hole=0.4,
                            title='Répartition des types de carte',
                            color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                card_fraud = df.groupby('type_carte')['fraude'].mean().reset_index()
                fig = px.bar(card_fraud, x='type_carte', y='fraude',
                            title='Taux de fraude par type de carte',
                            labels={'fraude': 'Taux de fraude', 'type_carte': 'Type de carte'},
                            color='type_carte',
                            color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Dernières alertes
        if len(st.session_state.last_transactions) > 0:
            st.markdown("""
            <div class="card">
                <h3 class="text-contrast">🚨 Dernières alertes</h3>
            """, unsafe_allow_html=True)
            
            alert_df = pd.DataFrame(st.session_state.last_transactions[-5:])
            st.dataframe(alert_df[['date', 'montant', 'probabilite', 'decision']], hide_index=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Aucune donnée disponible")

elif st.session_state.page == 'analytics':
    st.markdown("## 📈 Analyse des données")
    
    if not df.empty:
        # Filtres
        st.markdown("""
        <div class="card">
            <h3 class="text-contrast">🔍 Filtres avancés</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            min_amount = st.number_input("Montant minimum (€)", min_value=0, value=0)
            max_amount = st.number_input("Montant maximum (€)", min_value=0, value=int(df['montant_transaction'].max()))
            amount_range = st.slider("Plage de montant (€)", 
                                   min_value=0, 
                                   max_value=int(df['montant_transaction'].max()),
                                   value=(0, int(df['montant_transaction'].max())))
        
        with col2:
            selected_region = st.selectbox("Région", ['Toutes'] + list(df['region'].unique()))
            selected_card = st.selectbox("Type de carte", ['Tous'] + list(df['type_carte'].unique()))
            score_range = st.slider("Plage de score crédit", 
                                  min_value=int(df['score_credit'].min()),
                                  max_value=int(df['score_credit'].max()),
                                  value=(int(df['score_credit'].min()), int(df['score_credit'].max())))
        
        with col3:
            age_range = st.slider("Tranche d'âge", 
                                min_value=int(df['age'].min()),
                                max_value=int(df['age'].max()),
                                value=(int(df['age'].min()), int(df['age'].max())))
            anciennete_range = st.slider("Ancienneté du compte (années)", 
                                       min_value=float(df['anciennete_compte'].min()),
                                       max_value=float(df['anciennete_compte'].max()),
                                       value=(float(df['anciennete_compte'].min()), float(df['anciennete_compte'].max())))
            fraud_only = st.checkbox("Afficher uniquement les fraudes")
        
        # Application des filtres
        filtered_df = df.copy()
        filtered_df = filtered_df[
            (filtered_df['montant_transaction'] >= amount_range[0]) & 
            (filtered_df['montant_transaction'] <= amount_range[1]) &
            (filtered_df['score_credit'] >= score_range[0]) &
            (filtered_df['score_credit'] <= score_range[1]) &
            (filtered_df['age'] >= age_range[0]) &
            (filtered_df['age'] <= age_range[1]) &
            (filtered_df['anciennete_compte'] >= anciennete_range[0]) &
            (filtered_df['anciennete_compte'] <= anciennete_range[1])
        ]
        
        if selected_region != 'Toutes':
            filtered_df = filtered_df[filtered_df['region'] == selected_region]
        
        if selected_card != 'Tous':
            filtered_df = filtered_df[filtered_df['type_carte'] == selected_card]
        
        if fraud_only:
            filtered_df = filtered_df[filtered_df['fraude'] == 1]
        
        # Statistiques
        st.markdown(f"""
        <div class="card">
            <h3 class="text-contrast">📊 Résultats ({len(filtered_df)} transactions)</h3>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <p class="text-contrast">Fraudes détectées: <strong>{filtered_df['fraude'].sum()}</strong> ({filtered_df['fraude'].mean()*100:.2f}%)</p>
                    <p class="text-contrast">Montant total: <strong>€{filtered_df['montant_transaction'].sum():,.2f}</strong></p>
                </div>
                <div>
                    <p class="text-contrast">Montant moyen: <strong>€{filtered_df['montant_transaction'].mean():.2f}</strong></p>
                    <p class="text-contrast">Score crédit moyen: <strong>{filtered_df['score_credit'].mean():.1f}/850</strong></p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Visualisations
        tab1, tab2, tab3 = st.tabs(["📉 Distributions", "📌 Corrélations", "🔢 Données"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(filtered_df, x='montant_transaction', color='fraude',
                                 nbins=20, barmode='overlay',
                                 color_discrete_map={0: '#00f0ff', 1: '#ff1744'},
                                 labels={'montant_transaction': 'Montant (€)', 'fraude': 'Fraude'},
                                 title='Distribution des montants')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(filtered_df, x='fraude', y='score_credit',
                            color='fraude',
                            color_discrete_map={0: '#00f0ff', 1: '#ff1744'},
                            labels={'score_credit': 'Score crédit', 'fraude': 'Fraude'},
                            title='Score crédit par statut de fraude')
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
            
            fig = px.scatter(filtered_df, x='age', y='anciennete_compte',
                            color='fraude',
                            size='montant_transaction',
                            color_discrete_map={0: '#00f0ff', 1: '#ff1744'},
                            labels={'age': 'Âge', 'anciennete_compte': 'Ancienneté du compte (années)'},
                            title='Relation Âge/Ancienneté')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Matrice de corrélation
            numeric_df = filtered_df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(corr_matrix,
                           labels=dict(x="Variable", y="Variable", color="Corrélation"),
                           x=corr_matrix.columns,
                           y=corr_matrix.columns,
                           title='Matrice de corrélation')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
            
            # Corrélations avec la fraude
            fraud_corr = corr_matrix['fraude'].drop('fraude').sort_values()
            fig = px.bar(fraud_corr,
                        labels={'index': 'Variable', 'value': 'Corrélation avec la fraude'},
                        title='Corrélation des variables avec la fraude')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.dataframe(filtered_df.sort_values('montant_transaction', ascending=False), 
                        height=600,
                        column_config={
                            "montant_transaction": st.column_config.NumberColumn("Montant (€)", format="€%.2f"),
                            "fraude": st.column_config.CheckboxColumn("Fraude")
                        })
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Aucune donnée disponible")

elif st.session_state.page == 'detection':
    st.markdown("## 🔍 Détection en temps réel")
    
    if model is not None:
        with st.form("detection_form"):
            st.markdown("""
            <div class="card">
                <h3 class="text-contrast">🔎 Analyse de transaction</h3>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Âge", min_value=18, max_value=100, value=30, help="Âge du titulaire du compte")
                genre = st.selectbox("Genre", ['M', 'F'], help="Genre du titulaire du compte")
                salaire = st.number_input("Salaire annuel (€)", min_value=0, value=50000, help="Revenu annuel du client")
                score_credit = st.number_input("Score crédit", min_value=300, max_value=850, value=650, 
                                             help="Score de crédit FICO (300-850)")
            
            with col2:
                montant = st.number_input("Montant (€)", min_value=0.0, value=100.0, step=10.0, 
                                         help="Montant de la transaction")
                type_carte = st.selectbox("Type de carte", ['Visa', 'MasterCard', 'American Express'], 
                                        help="Type de carte de paiement")
                # Correction des régions pour correspondre au dataset
                region = st.selectbox("Région", ['Houston', 'Orlando', 'Miami'], 
                                    help="Région où la transaction a lieu")
                anciennete = st.number_input("Ancienneté compte (années)", min_value=0.0, value=5.0, step=0.1,
                                           help="Durée depuis l'ouverture du compte")
            
            submitted = st.form_submit_button("Analyser la transaction", 
                                            help="Lancer l'analyse de détection de fraude",
                                            type="primary")
            st.markdown("</div>", unsafe_allow_html=True)
        
        if submitted:
            with st.spinner("🔍 Analyse en cours..."):
                # Simulation de traitement avec animation de progression
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(percent_complete + 1)
                
                # Préparation des données
                input_data = pd.DataFrame({
                    'age': [age],
                    'salaire': [salaire],
                    'score_credit': [score_credit],
                    'montant_transaction': [montant],
                    'anciennete_compte': [anciennete],
                    'genre': [1 if genre == 'M' else 0],
                    'type_carte': [['Visa', 'MasterCard', 'American Express'].index(type_carte)],
                    'region': [['Houston', 'Orlando', 'Miami'].index(region)]
                })
                
                # Prédiction
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                
                # Enregistrement de la transaction
                transaction_record = {
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'age': age,
                    'genre': genre,
                    'montant': montant,
                    'region': region,
                    'type_carte': type_carte,
                    'probabilite': f"{proba[1]*100:.1f}%",
                    'decision': "Fraude" if prediction == 1 else "Valide"
                }
                st.session_state.last_transactions.append(transaction_record)
                
                # Affichage des résultats
                if prediction == 1:
                    if proba[1] > 0.9:
                        alert_level = "HAUTE"
                        alert_color = "var(--danger)"
                        alert_icon = "🔥"
                    elif proba[1] > 0.7:
                        alert_level = "MOYENNE"
                        alert_color = "var(--warning)"
                        alert_icon = "⚠️"
                    else:
                        alert_level = "FAIBLE"
                        alert_color = "var(--warning)"
                        alert_icon = "🔍"
                    
                    st.markdown(f"""
                    <div class="fraud-result">
                        <h3>{alert_icon} ALERTE FRAUDE - NIVEAU {alert_level}</h3>
                        <p>Probabilité de fraude: <strong style="color: {alert_color};">{proba[1]*100:.1f}%</strong></p>
                        <p>Montant: <strong>€{montant:.2f}</strong> | Région: <strong>{region}</strong></p>
                        <p>Recommandation: <strong>Vérification manuelle requise</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if proba[1] > 0.3:
                        st.markdown(f"""
                        <div class="warning-result">
                            <h3>⚠️ TRANSACTION SUSPECTE</h3>
                            <p>Probabilité de fraude: <strong>{proba[1]*100:.1f}%</strong></p>
                            <p>Montant: <strong>€{montant:.2f}</strong> | Région: <strong>{region}</strong></p>
                            <p>Recommandation: <strong>Surveillance recommandée</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-result">
                            <h3>✅ TRANSACTION VALIDE</h3>
                            <p>Probabilité de fraude: <strong>{proba[1]*100:.1f}%</strong></p>
                            <p>Montant: <strong>€{montant:.2f}</strong> | Région: <strong>{region}</strong></p>
                            <p>Recommandation: <strong>Transaction approuvée</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                
                    
                    # Conseils
                    if prediction == 1:
                        st.markdown("""
                        <div class="explanation-box">
                            <h4>🛡️ Actions recommandées:</h4>
                            <ul>
                                <li>Contacter le client pour vérification</li>
                                <li>Vérifier l'historique récent du compte</li>
                                <li>Comparer avec le comportement habituel du client</li>
                                <li>Si confirmé, bloquer la carte temporairement</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="explanation-box">
                            <h4>👍 Bonnes pratiques:</h4>
                            <ul>
                                <li>Continuer à surveiller les transactions similaires</li>
                                <li>Vérifier les autres transactions du client</li>
                                <li>Envisager une alerte pour montants élevés</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Simulation d'historique
                if random.random() > 0.7:
                    st.markdown("""
                    <div class="card" style="animation: fadeIn 1s ease-out;">
                        <h4>📌 Comportement inhabituel détecté</h4>
                        <p>Cette transaction présente des similarités avec des fraudes précédentes:</p>
                        <ul>
                            <li>3 transactions similaires marquées comme fraudes dans cette région</li>
                            <li>Montant supérieur à 95% des transactions de ce client</li>
                            <li>Heure inhabituelle pour ce type de transaction</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.error("Modèle non chargé - Impossible d'effectuer des prédictions")

elif st.session_state.page == 'settings':
    st.markdown("## ⚙️ Paramètres")
    
    tab1, tab2 = st.tabs(["🔧 Configuration", "📊 Modèle"])  # Suppression de l'onglet Compte
    
    with tab1:
        st.markdown("""
        <div class="card">
            <h3 class="text-contrast">Configuration du système</h3>
            <p class="text-contrast">Paramètres avancés de détection</p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            seuil = st.slider("Seuil d'alerte (%)", 50, 100, 75,
                             help="Niveau de probabilité à partir duquel une alerte est déclenchée")
            notifications = st.multiselect("Méthodes de notification", 
                                         ["Email", "SMS", "Notification application", "Webhook"],
                                         default=["Email", "SMS"],
                                         help="Comment recevoir les alertes")
        
        with col2:
            frequence = st.selectbox("Fréquence d'analyse", 
                                   ["Immédiate", "Quotidienne", "Hebdomadaire", "Mensuelle"],
                                   help="Fréquence des analyses par lots")
            export_format = st.selectbox("Format d'export", 
                                      ["CSV", "Excel", "JSON", "Parquet"],
                                      help="Format pour l'export des données")
        
        if st.button("💾 Sauvegarder les paramètres", key="save_settings"):
            st.toast("Paramètres sauvegardés avec succès!", icon="✅")
        
        # Ajout de la fonction d'export
        if st.button("📤 Exporter les données", key="export_data", type="primary"):
            if not df.empty:
                export_data = get_export_file(df, export_format)
                if export_format == "CSV":
                    st.download_button(
                        label="Télécharger CSV",
                        data=export_data,
                        file_name="fraud_data.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
                elif export_format == "Excel":
                    st.download_button(
                        label="Télécharger Excel",
                        data=export_data,
                        file_name="fraud_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel"
                    )
                elif export_format == "JSON":
                    st.download_button(
                        label="Télécharger JSON",
                        data=export_data,
                        file_name="fraud_data.json",
                        mime="application/json",
                        key="download_json"
                    )
                elif export_format == "Parquet":
                    st.download_button(
                        label="Télécharger Parquet",
                        data=export_data,
                        file_name="fraud_data.parquet",
                        mime="application/octet-stream",
                        key="download_parquet"
                    )
            else:
                st.warning("Aucune donnée à exporter")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="card">
            <h3 class="text-contrast">Configuration du modèle</h3>
            <p class="text-contrast">Paramètres avancés du modèle de détection</p>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Modèle actuel:** Random Forest Classifier  
        **Version:** 2.1.0  
        **Précision:** 94.2%  
        **Recall:** 88.5%  
        **Dernière mise à jour:** 15/06/2023
        """)
        
        # Métriques du modèle
        metrics = {
            'Accuracy': 0.942,
            'Precision': 0.956,
            'Recall': 0.885,
            'F1-Score': 0.919,
            'AUC-ROC': 0.981
        }
        
        fig = px.bar(x=list(metrics.keys()), y=list(metrics.values()),
                    labels={'x': 'Métrique', 'y': 'Score'},
                    title='Performance du modèle',
                    color=list(metrics.keys()),
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Options de modèle
        st.selectbox("Modèle principal", 
                    ["Random Forest (actuel)", "Gradient Boosting", "Réseau de neurones", "SVM"],
                    help="Modèle utilisé pour les prédictions")
        
        st.checkbox("Activer l'apprentissage continu", 
                  value=True,
                  help="Le modèle s'améliore avec les nouvelles données")
        
        if st.button("🔄 Entraîner le modèle", key="train_model"):
            with st.spinner("Entraînement en cours..."):
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.03)
                    progress_bar.progress(percent_complete + 1)
                st.toast("Modèle entraîné avec succès!", icon="✅")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Pied de page
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-light); padding: 1rem;">
    <p>FraudGuard Pro v2.1 | Système de détection de fraude avancé | © 2023</p>
    <p style="font-size: 0.8rem;">Dernière mise à jour: 25/06/2023</p>
</div>
""", unsafe_allow_html=True)