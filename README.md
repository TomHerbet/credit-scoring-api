# API de Scoring Crédit

Ce projet met en production un modèle de Machine Learning (LightGBM) permettant d'évaluer le risque de défaut de paiement d'un client bancaire. L'application est exposée via une API REST (FastAPI), conteneurisée avec Docker et suit un pipeline CI/CD complet.

## Objectif du Projet

L'objectif est de fournir aux chargés de clientèle un outil de décision en temps réel qui :
1.  **Calcule la probabilité de faillite** d'un client.
2.  **Prédit le statut** (Accepté/Refusé) selon un seuil de rentabilité optimisé.
3.  **Fournit des données contextuelles** (comparaison avec la moyenne des autres clients).

## Structure du Projet

Voici le détail de l'organisation des dossiers et fichiers :

* **`api_v4.py`** : Le cœur de l'application. Contient :
    * Le chargement du modèle et des données.
    * Le pipeline de *preprocessing* (nettoyage, encodage, features polynomiales).
    * Les endpoints de l'API (`/predict`, `/client`, etc.).
* **`Dockerfile`** : La "recette" pour construire l'image virtuelle du projet. Il définit l'environnement Python et installe les dépendances système nécessaires.
* **`docker-compose.yml`** : L'orchestrateur de déploiement. Il configure le service API et gère les liens avec les fichiers de données (volumes montés).
* **`.github/workflows/deploy.yml`** : Le pipeline d'automatisation (CI/CD).
    * *Test* : Lance les tests unitaires à chaque modification.
    * *Build* : Construit l'image Docker.
    * *Deploy* : Met à jour automatiquement le serveur VPS de production.
* **`tests/`** : Dossier contenant les tests unitaires (`test_api.py`) pour vérifier le bon fonctionnement de l'API sans charger les gros fichiers de données (utilisation de "mocks").
* **`requirements.txt`** : Liste de toutes les librairies Python nécessaires au fonctionnement.
* **`.gitignore`** : Liste des fichiers à exclure du versionning (modèles lourds .pkl, données .csv, dossiers virtuels, etc.).

---
*Note : Les fichiers de données (`application_train/test.csv`) et le modèle (`best_lgbm_model.pkl`) ne sont pas dans le dépôt pour des raisons de taille, mais sont requis à la racine du VPS pour l'exécution.*

==========================================
# API & Web Server (FastAPI / Flask)
==========================================
fastapi==0.120.1
uvicorn==0.38.0
starlette==0.49.0
requests==2.32.5
httpx
Flask==3.1.2
flask-cors==6.0.1
waitress==3.0.2
Werkzeug==3.1.3
graphene==3.4.3
graphql-core==3.2.6
graphql-relay==3.2.0
anyio==4.11.0
h11==0.16.0
urllib3==2.5.0

==========================================
# Machine Learning & AI
==========================================
lightgbm==4.6.0
xgboost==3.1.1
catboost==1.2.8
scikit-learn==1.7.2
imbalanced-learn==0.14.0
imblearn==0.0
shap
joblib==1.5.2
threadpoolctl==3.6.0

==========================================
# Data Processing & Analysis
==========================================
pandas==2.3.3
numpy==2.3.4
scipy==1.16.2
pyarrow==21.0.0
narwhals==2.10.0

==========================================
# Visualization
==========================================
matplotlib==3.10.7
seaborn==0.13.2
plotly==6.3.1
missingno==0.5.2
pillow==12.0.0
graphviz==0.21
contourpy==1.3.3
cycler==0.12.1
fonttools==4.60.1
kiwisolver==1.4.9

==========================================
# MLOps, Tracking & DevOps
==========================================
mlflow==3.5.1
mlflow-skinny==3.5.1
mlflow-tracing==3.5.1
databricks-sdk==0.70.0
opentelemetry-api==1.38.0
opentelemetry-proto==1.38.0
opentelemetry-sdk==1.38.0
opentelemetry-semantic-conventions==0.59b0
docker==7.1.0
GitPython==3.1.45
gitdb==4.0.12
smmap==5.0.2

==========================================
# Database & Migrations
==========================================
SQLAlchemy==2.0.44
alembic==1.17.0
Mako==1.3.10
sqlparse==0.5.3

==========================================
# Utilities & Core Dependencies
==========================================
pydantic==2.12.3
pydantic_core==2.41.4
python-dotenv==1.2.1
PyYAML==6.0.3
click==8.3.0
colorama==0.4.6
packaging==25.0
six==1.17.0
zipp==3.23.0
typing-inspection==0.4.2
typing_extensions==4.15.0
importlib_metadata==8.7.0
annotated-doc==0.0.3
annotated-types==0.7.0
python-dateutil==2.9.0.post0
pytz==2025.2
tzdata==2025.2
cachetools==6.2.1
blinker==1.9.0
greenlet==3.2.4
cloudpickle==3.1.1
pycparser==2.23
cffi==2.0.0
charset-normalizer==3.4.4
idna==3.11
Jinja2==3.1.6
MarkupSafe==3.0.3
pyparsing==3.2.5

==========================================
# Security & Auth
==========================================
cryptography==46.0.3
rsa==4.9.1
certifi==2025.10.5
google-auth==2.41.1
itsdangerous==2.2.0
pyasn1==0.6.1
pyasn1_modules==0.4.2
