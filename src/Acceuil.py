import streamlit as st

def main():
    st.set_page_config(
        page_title="Prédiction du Churn Client",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .header {
            text-align: center;
            color: #4CAF50;
            font-size: 2.5em;
            font-weight: bold;
        }
        .section-title {
            color: #FF5733;
            font-size: 1.5em;
            font-weight: bold;
        }
        .instructions {
            color: #2E86C1;
            font-size: 1.2em;
        }
        .list-item {
            color: #8E44AD;
            font-size: 1.1em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h1 class='header'>Prédiction du Churn Client 🛍️</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class='section-title'>Bienvenue sur notre application de prédiction du churn client !</div>

        <div class='instructions'>
        Cette application vous permet d'explorer les données, de réaliser des prédictions individuelles, et d'analyser des prédictions en lot.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class='section-title'>Naviguez entre les sections :</div>

        <ul>
            <li class='list-item'>📊 <b>Dashboard Exploratoire</b> : Analysez les données et découvrez des insights.</li>
            <li class='list-item'>🔍 <b>Prédiction Individuelle</b> : Prédisez si un client spécifique va se désabonner.</li>
            <li class='list-item'>📋 <b>Prédictions en Lot</b> : Génération de clients fictifs et analyses prédictives.</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        """
        <div class='section-title'>Comment utiliser l'application :</div>

        <div class='instructions'>
        <ul>
            <li>Utilisez la barre latérale pour naviguer entre les pages.</li>
            <li>Suivez les instructions sur chaque page pour interagir avec les fonctionnalités.</li>
            <li>Si vous avez des questions, n'hésitez pas à nous contacter via la page <b>À Propos</b>.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
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
