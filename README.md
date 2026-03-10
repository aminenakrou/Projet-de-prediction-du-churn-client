# **📊 Prédiction du Churn Client**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aminenakrou-churn-prediction.streamlit.app)

> 👆 **Cliquez sur le badge pour accéder au dashboard en ligne !**

Bienvenue dans l'application de prédiction du churn client ! Ce projet utilise des techniques de data science et de machine learning pour analyser les données clients et prédire les risques de churn (désabonnement). L'objectif est de fournir une analyse approfondie pour identifier les clients à risque et permettre aux entreprises de mettre en place des stratégies de rétention efficaces.



## **📚 Table des Matières**

- [📖 Contexte](#-contexte)
- [🎯 Objectifs](#-objectifs)
- [✨ Fonctionnalités](#-fonctionnalités)
- [📘 Documentation Complète](#-documentation-complète)
- [🏗️ Structure du Dépôt](#️-structure-du-dépôt)
- [🧠 Mise en Place du Modèle Prédictif](#-mise-en-place-du-modèle-prédictif)
- [⚙️ Installation](#️-installation)
- [🚀 Utilisation](#-utilisation)
- [🐳 Docker](#-docker)
- [👤 Auteur](#-auteur)



## **📖 Contexte**

Le churn client, ou attrition client, représente un enjeu majeur pour les entreprises en termes de fidélisation. Comprendre pourquoi les clients se désabonnent est crucial pour optimiser la satisfaction client et minimiser les pertes financières.



## **🎯 Objectifs**

L'application a pour objectifs de :

- **Analyser les données clients** pour identifier les facteurs contribuant au churn.
- **Prédire les clients à risque** grâce à des modèles de machine learning performants.
- **Fournir des visualisations interactives** pour explorer les données et les résultats prédictifs.
- **Proposer des prédictions individuelles et en lot** pour couvrir différents besoins.



## **📘 Documentation Complète**

Une documentation exhaustive est disponible dans le dossier [`docs/`](docs/) pour vous aider à maîtriser tous les aspects du projet :

### 📚 Guides Disponibles

1. **[Fondamentaux du Machine Learning](docs/01_machine_learning_fundamentals.md)**
   - Concepts de base du ML
   - Types d'apprentissage
   - Workflow typique
   - Bonnes pratiques

2. **[Types de Tâches en ML](docs/02_ml_task_types.md)**
   - Régression, Classification
   - Clustering, Détection d'anomalies
   - Séries temporelles
   - Comment choisir la bonne approche

3. **[Guide Complet des Métriques](docs/03_metrics_guide.md)**
   - Métriques de régression et classification
   - Quand utiliser quelle métrique
   - Combiner plusieurs métriques
   - Métriques business vs ML

4. **[Guide Churn Prediction](docs/04_churn_prediction_guide.md)**
   - Qu'est-ce que le churn ?
   - Impact business et ROI
   - Création d'un modèle de prédiction
   - Stratégies de rétention

5. **[Meilleures Pratiques EDA](docs/05_eda_best_practices.md)**
   - Analyse exploratoire des données
   - Visualisations recommandées
   - Feature engineering
   - Checklist complète

6. **[Déploiement et Recommandations](docs/06_model_deployment_recommendations.md)**
   - Du notebook à la production
   - Création d'une API
   - Docker et CI/CD
   - Monitoring en production

> 💡 **Conseil** : Consultez le [README de la documentation](docs/README.md) pour un guide complet !



## **✨ Fonctionnalités**

### **1. Dashboard Exploratoire 📊**

- Visualisation des données brutes et statistiques descriptives.
- Graphiques interactifs pour analyser les distributions et les relations avec le churn.
- Comparaison des services et des caractéristiques clients.

### **2. Prédiction Individuelle 🔍**

- Interface permettant d'entrer les informations d'un client spécifique.
- Prédiction du risque de churn avec explication.

### **3. Prédictions en Lot 🧮**

- Génération de clients fictifs.
- Prédiction du churn pour un grand nombre de clients en une seule opération.
- Visualisation des résultats sous forme de graphiques interactifs.

### **4. Page À Propos 👤**

- Présentation de l'auteur et contact.



## **🏗️ Structure du Dépôt**

Le projet est organisé comme suit :

```
Churn-Prediction-and-Analysis-Project/
│
├── data/
│   └── data.csv                # Données utilisées pour le projet
│
├── docs/                       # 📚 Documentation complète
│   ├── 01_machine_learning_fundamentals.md
│   ├── 02_ml_task_types.md
│   ├── 03_metrics_guide.md
│   ├── 04_churn_prediction_guide.md
│   ├── 05_eda_best_practices.md
│   ├── 06_model_deployment_recommendations.md
│   └── README.md
│
├── notebooks/
│   └── Train-Models.ipynb      # Jupyter Notebook pour l'entraînement des modèles
│
├── src/
│   ├── Acceuil.py              # Application Streamlit principale (page d'accueil)
│   ├── pages/
│   │   ├── 1_📊_Dashboard_Exploratoire.py
│   │   ├── 2_🔍_Prédiction_Individuelle.py
│   │   ├── 3_🧮_Prédictions_en_Lot.py
│   │   └── 4_👤_À_Propos.py
│   └── models/
│       └── model_LogisticRegression.pkl  # Modèle entraîné
│
├── Dockerfile                  # 🐳 Configuration Docker
├── .dockerignore
├── requirements.txt
└── README.md
```



## **🧠 Mise en Place du Modèle Prédictif**

La mise en place du modèle prédictif a été réalisée de la manière suivante :

### **1. Chargement des Données**

Les données clients ont été chargées à partir du fichier `data/data.csv`. Elles contiennent des informations telles que :

- **Identifiant du client**, **genre**, **ancienneté**, **services souscrits**, **méthode de paiement**, **frais mensuels**, etc.
- **Churn** (cible) indiquant si le client a quitté l'entreprise.

### **2. Prétraitement des Données**

- **Nettoyage des Données** : Gestion des valeurs manquantes et conversion des types de données.
- **Encodage des Variables Catégorielles** : Utilisation de techniques d'encodage adaptées (Label Encoding et One-Hot Encoding).
- **Normalisation des Données** : Normalisation des variables numériques pour améliorer la performance du modèle.

### **3. Analyse Exploratoire (EDA)**

- **Analyse Univariée et Bivariée** pour comprendre la distribution des variables et leur relation avec le churn.
- **Visualisations** pour identifier les tendances et les schémas pertinents.

### **4. Création et Entraînement du Modèle**

- **Modèles Testés** :
  - Régression Logistique
  - Random Forest
  - SVM (Support Vector Machine)
- **Pipeline Scikit-learn** : Mise en place d'un pipeline comprenant le prétraitement et l'entraînement du modèle.
- **Validation Croisée** : Utilisation de la validation croisée pour évaluer les performances du modèle.

### **5. Évaluation du Modèle**

- **Métriques Utilisées** :
  - Précision
  - Rappel
  - F1-Score
  - AUC-ROC
- **Meilleur Modèle** : Le modèle final sélectionné est une régression logistique, sauvegardée sous `src/models/model_LogisticRegression.pkl`.

Pour plus de détails, consultez le notebook `notebooks/Model_Building.ipynb`.



## **⚙️ Installation**

### **Prérequis**

- **Python 3.7+** installé.
- **Pip** pour gérer les packages Python.

### **Étapes d'Installation**

1. **Clonez le projet :**

   ```bash
   git clone https://github.com/aminenakrou/Churn-Prediction-and-Analysis-Project.git
   cd Churn-Prediction-and-Analysis-Project
   ```

2. **Créez un environnement virtuel :**

   ```bash
   python -m venv env
   ```

3. **Activez l'environnement virtuel :**

   - Sur Windows :

     ```bash
     .\env\Scripts\activate
     ```

   - Sur MacOS/Linux :

     ```bash
     source env/bin/activate
     ```

4. **Installez les dépendances :**

   ```bash
   pip install -r requirements.txt
   ```



## **🚀 Utilisation**

1. **Lancez l'application Streamlit :**

   ```bash
   streamlit run src/Acceuil.py

   ```

2. **Naviguez à travers l'application :**

   - Utilisez la barre latérale pour accéder aux différentes pages.
   - Explorez les visualisations, effectuez des prédictions individuelles ou en lot.



## **☁️ Déploiement sur Streamlit Cloud**

Pour mettre l'application en ligne gratuitement et obtenir un lien public :

1. **Pushez ce dépôt sur votre GitHub** (`github.com/aminenakrou`)
2. **Allez sur** [share.streamlit.io](https://share.streamlit.io) et connectez votre compte GitHub
3. **Cliquez sur "New app"** et configurez :
   - **Repository** : `aminenakrou/Churn-Prediction-and-Analysis-Project`
   - **Branch** : `main`
   - **Main file path** : `src/Acceuil.py`
4. **Cliquez "Deploy"** — l'URL générée sera de la forme `https://xxxxx.streamlit.app`
5. **Mettez à jour le badge** dans ce README en remplaçant l'URL dans la ligne `[![Streamlit App]...](...)`



## **🐳 Docker**

L'application peut être facilement déployée avec Docker pour une portabilité maximale.

### **Utilisation avec Docker**

1. **Construire l'image Docker :**

   ```bash
   docker build -t churn-prediction-app .
   ```

2. **Lancer le conteneur :**

   ```bash
   docker run -p 8501:8501 churn-prediction-app
   ```

3. **Accéder à l'application :**

   Ouvrez votre navigateur à l'adresse : `http://localhost:8501`

### **Utilisation avec Docker Compose** (Optionnel)

```bash
docker-compose up
```

### **CI/CD**

Le projet inclut une configuration GitHub Actions pour :
- ✅ Vérifier que le build Docker fonctionne
- ✅ Tester l'application automatiquement
- ✅ Assurer la qualité du code

Consultez `.github/workflows/` pour plus de détails.



## **👤 Auteur**

### ** AMINE NAKROU **

- **Élève-ingénieur Informatique, Statistiques & IA Polytech Lille — ISIA 4e année**
- Passionné par les sciences de données et l'intelligence artificielle.
- **Email** : [aminenakrou635@gmail.com]
- **WhatsApp** : +33 7 44 28 00 10 
- **LinkedIn** : [https://www.linkedin.com/in/amine-nakrou/]



**🎉 Merci d'utiliser l'application de prédiction du churn client !** N'oubliez pas de laisser une étoile ⭐ sur le dépôt si vous avez trouvé le projet utile.

