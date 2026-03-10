import streamlit as st
import pandas as pd
import joblib
import os
from pydantic import BaseModel, field_validator
from typing import Optional, Literal


# Charger le modèle entraîné (chemin absolu, robuste sur Streamlit Cloud)
_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'model_LogisticRegression.pkl')
model = joblib.load(_model_path)


# Définir la classe CustomerData avec Pydantic
class CustomerData(BaseModel):
    gender: Literal['Male', 'Female']
    SeniorCitizen: int  # 0 ou 1
    Partner: Literal['Yes', 'No']
    Dependents: Literal['Yes', 'No']
    tenure: int
    PhoneService: Literal['Yes', 'No']
    MultipleLines: Literal['Yes', 'No', 'No phone service']
    InternetService: Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity: Literal['Yes', 'No', 'No internet service']
    OnlineBackup: Literal['Yes', 'No', 'No internet service']
    DeviceProtection: Literal['Yes', 'No', 'No internet service']
    TechSupport: Literal['Yes', 'No', 'No internet service']
    StreamingTV: Literal['Yes', 'No', 'No internet service']
    StreamingMovies: Literal['Yes', 'No', 'No internet service']
    Contract: Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling: Literal['Yes', 'No']
    PaymentMethod: Literal[
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ]
    MonthlyCharges: float
    TotalCharges: Optional[float]  # Peut être manquant

    @field_validator('SeniorCitizen')
    def senior_citizen_must_be_0_or_1(cls, v):
        if v not in [0, 1]:
            raise ValueError('SeniorCitizen must be 0 ou 1')
        return v

    @field_validator('tenure')
    def tenure_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('tenure must be non-negative')
        return v

    @field_validator('MonthlyCharges', 'TotalCharges')
    def charges_must_be_positive(cls, v, info):
        if v is not None and v < 0:
            raise ValueError(f'{info.field_name} must be non-negative')
        return v

def predict_churn(customer_data: CustomerData):
    """
    Prend une instance de CustomerData, la transforme en DataFrame,
    et effectue une prédiction en utilisant le modèle chargé.
    """
    # Conversion en DataFrame
    data_dict = customer_data.model_dump()
    df = pd.DataFrame([data_dict])

    # Gérer les valeurs manquantes pour 'TotalCharges'
    if df['TotalCharges'].isnull().any():
        df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']

    # Effectuer la prédiction
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    # Interprétation du résultat
    churn_label = 'Yes' if prediction[0] == 1 else 'No'
    churn_probability = prediction_proba[0][1]

    return {
        'Churn': churn_label,
        'Probability': churn_probability
    }

# Construire l'application Streamlit
def main():
    st.set_page_config(
        page_title="Prédiction du Churn Client",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <h1 style='text-align: center; color: #4CAF50;'>Prédiction du Churn Client 📉</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h3 style='text-align: center; color: #f68b28;'>Prédisez si un client va se désabonner !</h3>
        """,
        unsafe_allow_html=True
    )

    st.write("Veuillez entrer les informations du client ci-dessous :")

    # Utiliser un expander pour les champs d'entrée
    with st.expander("Informations du Client 📝", expanded=True):
        # Créer des conteneurs pour les champs d'entrée
        input_container = st.container()

        # Disposition en colonnes
        col1, col2, col3 = input_container.columns(3)

        with col1:
            gender = st.selectbox("Genre", ['Male', 'Female'])
            SeniorCitizen = st.selectbox("Citoyen Senior", [0, 1])
            Partner = st.selectbox("Partenaire", ['Yes', 'No'])
            Dependents = st.selectbox("Personnes à charge", ['Yes', 'No'])
            tenure = st.number_input("Ancienneté (mois)", min_value=0, max_value=100, value=12)
            PhoneService = st.selectbox("Service Téléphonique", ['Yes', 'No'])
            MultipleLines = st.selectbox("Lignes Multiples", ['Yes', 'No', 'No phone service'])
            InternetService = st.selectbox("Service Internet", ['DSL', 'Fiber optic', 'No'])

        with col2:
            OnlineSecurity = st.selectbox("Sécurité en ligne", ['Yes', 'No', 'No internet service'])
            OnlineBackup = st.selectbox("Sauvegarde en ligne", ['Yes', 'No', 'No internet service'])
            DeviceProtection = st.selectbox("Protection de l'appareil", ['Yes', 'No', 'No internet service'])
            TechSupport = st.selectbox("Support Technique", ['Yes', 'No', 'No internet service'])
            StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
            StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
            Contract = st.selectbox("Contrat", ['Month-to-month', 'One year', 'Two year'])
            PaperlessBilling = st.selectbox("Facturation sans papier", ['Yes', 'No'])

        with col3:
            PaymentMethod = st.selectbox("Méthode de paiement", [
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
            ])
            MonthlyCharges = st.number_input("Charges Mensuelles", min_value=0.0, value=0.0)
            TotalCharges = st.number_input("Charges Totales (optionnel)", min_value=0.0, value=0.0)

    # Personnalisation du bouton avec HTML et CSS
    button_style = """
    <style>
    div.stButton > button:first-child {
        background-color: #f68b28;
        color:#ffffff;
        font-size:20px;
        height:3em;
        width:10em;
        border-radius:10px;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff9933;
        color:#ffffff;
    }
    </style>
    """

    st.markdown(button_style, unsafe_allow_html=True)

    # Bouton de prédiction
    predict_button = st.button("Prédire 📈")

    if predict_button:
        # Préparer les données d'entrée
        customer_input = {
            'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'tenure': tenure,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges if TotalCharges != 0 else None
        }

        # Validation et prédiction
        try:
            customer_data = CustomerData(**customer_input)
            result = predict_churn(customer_data)

            # Afficher les résultats
            st.markdown("---")
            st.markdown(
                "<h2 style='text-align: center; color: #4CAF50;'>Résultats de la Prédiction</h2>",
                unsafe_allow_html=True
            )

            # CSS personnalisé pour les métriques avec couleurs dynamiques
            st.markdown(
                """
                <style>
                div.metric-container {
                    background-color: #f0f2f6;
                    border: 1px solid #e1e1e1;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    color: #333333;
                }
                div.metric-container.positive {
                    background-color: #d1e7dd;
                }
                div.metric-container.negative {
                    background-color: #f8d7da;
                }
                div.metric-container h3 {
                    font-size: 2em;
                    margin: 0;
                }
                div.metric-container p {
                    font-size: 1.5em;
                    margin: 0;
                    font-weight: bold;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Déterminer la classe CSS en fonction du résultat
            churn_class = 'negative' if result['Churn'] == 'Yes' else 'positive'
            churn_emoji = '🚨' if result['Churn'] == 'Yes' else '✅'

            # Afficher les métriques dans des colonnes
            with st.container():
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(
                        f"""
                        <div class='metric-container {churn_class}'>
                            <h3>Churn Prédit {churn_emoji}</h3>
                            <p>{result['Churn']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:
                    st.markdown(
                        f"""
                        <div class='metric-container'>
                            <h3>Probabilité 📊</h3>
                            <p>{result['Probability']*100:.2f}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.error(f"Erreur dans les données d'entrée : {e}")
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
