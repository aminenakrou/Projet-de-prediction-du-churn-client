import streamlit as st

def main():
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>À Propos de l'Auteur 📝</h1>",
        unsafe_allow_html=True
    )

    # Ajout d'un expander pour l'auteur
    with st.expander("Auteur", True):
        st.header("**Amine NAKROU**")
        st.markdown("""
            *:blue[Élève-ingénieur Informatique, Statistiques & IA — Polytech Lille (ISIA 4e année)]*
            Bonjour,
            Je suis Amine, élève-ingénieur spécialisé en Informatique, Statistiques et Intelligence Artificielle à Polytech Lille.
            Passionné par la data science et le machine learning, je m'engage à concevoir des solutions analytiques claires et performantes.

            * **Email** : [aminenakrou635@gmail.com](mailto:aminenakrou635@gmail.com)
            * **WhatsApp** : +33 7 44 28 00 10
            * **LinkedIn** : [Amine NAKROU](https://www.linkedin.com/in/amine-nakrou/)
            * **GitHub** : [aminenakrou](https://github.com/aminenakrou)
        """, unsafe_allow_html=True)

    # Sidebar content
    with st.sidebar:
        st.markdown("""
            ## Auteur
            *:blue[Amine NAKROU]*
            * **Email** : [aminenakrou635@gmail.com](mailto:aminenakrou635@gmail.com)
            * **WhatsApp** : +33 7 44 28 00 10
            * **LinkedIn** : [Amine NAKROU](https://www.linkedin.com/in/amine-nakrou/)
            * **GitHub** : [aminenakrou](https://github.com/aminenakrou)
        """, unsafe_allow_html=True)
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
