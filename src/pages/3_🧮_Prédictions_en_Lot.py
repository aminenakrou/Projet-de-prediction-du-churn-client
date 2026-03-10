import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# Charger le modèle entraîné (chemin absolu, robuste sur Streamlit Cloud)
_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'model_LogisticRegression.pkl')
model = joblib.load(_model_path)

# Définir les options possibles pour les variables catégorielles
categorical_variables = {
    'Genre': ['gender', ['Male', 'Female']],
    'Citoyen Senior': ['SeniorCitizen', [0, 1]],
    'Partenaire': ['Partner', ['Yes', 'No']],
    'Personnes à charge': ['Dependents', ['Yes', 'No']],
    'Service Téléphonique': ['PhoneService', ['Yes', 'No']],
    'Lignes Multiples': ['MultipleLines', ['Yes', 'No', 'No phone service']],
    'Service Internet': ['InternetService', ['DSL', 'Fiber optic', 'No']],
    'Sécurité en ligne': ['OnlineSecurity', ['Yes', 'No', 'No internet service']],
    'Sauvegarde en ligne': ['OnlineBackup', ['Yes', 'No', 'No internet service']],
    'Protection de l\'appareil': ['DeviceProtection', ['Yes', 'No', 'No internet service']],
    'Support Technique': ['TechSupport', ['Yes', 'No', 'No internet service']],
    'Streaming TV': ['StreamingTV', ['Yes', 'No', 'No internet service']],
    'Streaming Movies': ['StreamingMovies', ['Yes', 'No', 'No internet service']],
    'Contrat': ['Contract', ['Month-to-month', 'One year', 'Two year']],
    'Facturation sans papier': ['PaperlessBilling', ['Yes', 'No']],
    'Méthode de paiement': ['PaymentMethod', [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ]]
}

numerical_variables = {
    'Ancienneté (mois)': ['tenure', 0, 72],
    'Charges Mensuelles': ['MonthlyCharges', 18.0, 118.0],
}

# Fonction pour générer des clients aléatoires
@st.cache_data
def generate_random_customers(n, seed):
    np.random.seed(seed)
    data = {}
    for var_name, (var_key, options) in categorical_variables.items():
        data[var_key] = np.random.choice(options, n)
    for var_name, (var_key, min_val, max_val) in numerical_variables.items():
        if isinstance(min_val, int):
            data[var_key] = np.random.randint(min_val, max_val + 1, n)
        else:
            data[var_key] = np.random.uniform(min_val, max_val, n)
    df = pd.DataFrame(data)
    df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
    return df

# Fonction pour prédire en lot
@st.cache_data
def predict_bulk(df):
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    df['Churn_Prediction'] = predictions
    df['Churn_Prediction_Label'] = df['Churn_Prediction'].map({0: 'No', 1: 'Yes'})
    df['Churn_Probability'] = probabilities
    return df

def main():
    st.set_page_config(
        page_title="Prédictions en Lot",
        page_icon="📈",
        layout="wide",
    )

    st.markdown(
        """
        <h1 style='text-align: center; color: #4CAF50;'>Prédictions en Lot 🧮</h1>
        """,
        unsafe_allow_html=True
    )
    st.write("Cette page génère aléatoirement des clients fictifs et prédit s'ils vont se désabonner.")

    with st.expander("Paramètres de Génération des Données ⚙️", expanded=False):
        n_customers = st.slider("Nombre de clients à générer", min_value=100, max_value=50000, value=1000, step=100)
        random_seed = st.number_input("Seed aléatoire", min_value=0, max_value=1000000, value=42, step=1)
    
    # Génération et prédiction sans bouton
    with st.spinner('Génération des données et prédictions en cours...'):
        # Générer les données
        df_customers = generate_random_customers(n_customers, random_seed)
        # Prédire le churn
        df_results = predict_bulk(df_customers)

    # Afficher quelques exemples
    with st.expander("Exemples de Données Générées", expanded=False):
        st.subheader("Aperçu des données générées")
        st.dataframe(df_results.head())

    # Afficher les résultats globaux
    churn_counts = df_results['Churn_Prediction_Label'].value_counts()
    st.markdown(
        """
        <h2 style='text-align: center; font-weight: bold; color: #FF5733;'>Résultats des Prédictions</h2>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <p>Sur les <strong>{n_customers}</strong> clients générés :</p>
            <p style='color: red;'>🔴 <strong>{churn_counts.get('Yes', 0)}</strong> vont potentiellement se désabonner.</p>
            <p style='color: green;'>🟢 <strong>{churn_counts.get('No', 0)}</strong> vont potentiellement rester.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Visualisations
    st.subheader("Visualisations")
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container(border=True):
                # Graphique en secteurs du churn avec Plotly
                fig1 = px.pie(
                    df_results,
                    names='Churn_Prediction_Label',
                    title='Répartition du Churn',
                    #color_discrete_sequence=px.colors.qualitative.Set1
                )
                st.plotly_chart(fig1, use_container_width=True)
        with col2:
            with st.container(border=True):
                selected_categorical_var = st.selectbox(
                    "Choisissez une variable catégorielle pour l'analyse",
                    list(categorical_variables.keys()),
                    index=13  # Par défaut sur 'Contrat'
                )
                # Churn par variable catégorielle sélectionnée
                var_key = categorical_variables[selected_categorical_var][0]
                df_grouped = df_results.groupby([var_key, 'Churn_Prediction_Label']).size().reset_index(name='Count')
                fig2 = px.bar(
                    df_grouped,
                    x=var_key,
                    y='Count',
                    color='Churn_Prediction_Label',
                    title=f'Churn par {selected_categorical_var}',
                    barmode='stack',
                    #color_discrete_sequence=px.colors.qualitative.Set1
                )
                st.plotly_chart(fig2, use_container_width=True)

    # Option pour télécharger les résultats
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger les résultats en CSV 📥",
        data=csv,
        file_name='predictions_churn.csv',
        mime='text/csv',
    )
    # Footer avec emojis et style
    st.markdown("---")
    st.markdown(
        """
        <style>
        .footer {
            text-align: center;
            color: #4CAF50;
            font-size: 16px;
            margin-top: 20px;
        }
        .footer a {
            color: #4CAF50;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        </style>
        <p class='footer'>💡 Dashboard créé avec ❤️ par <a href='https://github.com/aminenakrou' target='_blank'>Amine NAKROU</a></p>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
