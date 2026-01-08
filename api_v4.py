import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import shap
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# ===============================================================
# Configuration générale
# ===============================================================
app = FastAPI(title="Credit Risk API", version="1.0")

# --- CORRECTION DES CHEMINS POUR DOCKER ---
# On utilise des chemins absolus pointant vers /app/ car c'est là que le conteneur travaille
MODEL_PATH = "best_lgbm_model.pkl"

# On suppose que les données sont aussi à la racine /app/ (ou montées via volumes)
DATA_DIR = "."

model = None
preprocessed_test_data = None
test_ids = None
optimal_threshold = 0.4779
# Stockage des données brutes pour les infos clients
raw_application_train = None
raw_application_test = None


class PredictionResponse(BaseModel):
    sk_id_curr: int
    probability: float
    prediction: int


class ClientInfoResponse(BaseModel):
    sk_id_curr: int
    client_data: Dict[str, Any]


class GeneralInfoResponse(BaseModel):
    nb_credits: int
    revenu_moyen: float
    credit_moyen: float


# ===============================================================
# Fonctions de chargement
# ===============================================================
def load_model(model_path=None):
    if model_path is None:
        model_path = MODEL_PATH
    try:
        print(f"Tentative de chargement du modèle depuis: {model_path}")
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        print(f"✓ Modèle chargé avec succès")
        return loaded_model
    except Exception as e:
        print(f"✗ Erreur lors du chargement du modèle: {e}")
        raise


def load_data(train_path=None, test_path=None):
    if train_path is None:
        train_path = os.path.join(DATA_DIR, 'application_train.csv')
    if test_path is None:
        test_path = os.path.join(DATA_DIR, 'application_test.csv')
    
    try:
        print(f"Tentative de chargement des données depuis: {DATA_DIR}")
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        print(f"✓ Données chargées - Train: {train.shape}, Test: {test.shape}")
        return train, test
    except Exception as e:
        print(f"✗ Erreur lors du chargement des données: {e}")
        print(f"Chemins testés: Train={train_path}, Test={test_path}")
        raise


# ===============================================================
# Fonctions pour infos clients
# ===============================================================
def load_infos_gen(data):
    """
    Charge les informations générales sur l'ensemble des clients
    """
    lst_infos = [
        data.shape[0],
        round(data["AMT_INCOME_TOTAL"].mean(), 2),
        round(data["AMT_CREDIT"].mean(), 2)
    ]
    nb_credits = lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]
    
    return nb_credits, rev_moy, credits_moy


def identite_client(data, sk_id_curr):
    """
    Retourne les informations d'identité d'un client spécifique
    """
    data_client = data[data['SK_ID_CURR'] == int(sk_id_curr)]
    
    if data_client.empty:
        return None
    
    # Convertir en dictionnaire pour faciliter l'envoi JSON
    client_dict = data_client.iloc[0].to_dict()
    
    # Remplacer les valeurs NaN par None pour la sérialisation JSON
    client_dict = {k: (None if pd.isna(v) else v) for k, v in client_dict.items()}
    
    return client_dict


def get_client_summary(data, sk_id_curr):
    """
    Retourne un résumé des informations clés du client
    """
    client_data = identite_client(data, sk_id_curr)
    
    if client_data is None:
        return None
    
    # Sélectionner les informations les plus pertinentes
    summary = {
        'SK_ID_CURR': client_data.get('SK_ID_CURR'),
        'CODE_GENDER': client_data.get('CODE_GENDER'),
        'DAYS_BIRTH': client_data.get('DAYS_BIRTH'),
        'AGE': abs(client_data.get('DAYS_BIRTH', 0)) // 365 if client_data.get('DAYS_BIRTH') else None,
        'NAME_FAMILY_STATUS': client_data.get('NAME_FAMILY_STATUS'),
        'CNT_CHILDREN': client_data.get('CNT_CHILDREN'),
        'NAME_EDUCATION_TYPE': client_data.get('NAME_EDUCATION_TYPE'),
        'NAME_INCOME_TYPE': client_data.get('NAME_INCOME_TYPE'),
        'NAME_CONTRACT_TYPE': client_data.get('NAME_CONTRACT_TYPE'),
        'AMT_INCOME_TOTAL': client_data.get('AMT_INCOME_TOTAL'),
        'AMT_CREDIT': client_data.get('AMT_CREDIT'),
        'AMT_ANNUITY': client_data.get('AMT_ANNUITY'),
        'DAYS_EMPLOYED': client_data.get('DAYS_EMPLOYED'),
        'OCCUPATION_TYPE': client_data.get('OCCUPATION_TYPE'),
        'ORGANIZATION_TYPE': client_data.get('ORGANIZATION_TYPE')
    }
    
    return summary


# ===============================================================
# Préprocessing des données
# ===============================================================
def preprocess_data(application_train, application_test):
    """
    Preprocessing identique au notebook
    """
    print("Début du preprocessing...")
    
    # Créer la colonne AGE
    application_train['AGE'] = (application_train['DAYS_BIRTH'] / -365).astype(int)
    application_test['AGE'] = (application_test['DAYS_BIRTH'] / -365).astype(int)
    
    # Create an anomalous flag column
    application_train['DAYS_EMPLOYED_ANOM'] = application_train["DAYS_EMPLOYED"] == 365243
    application_test['DAYS_EMPLOYED_ANOM'] = application_test["DAYS_EMPLOYED"] == 365243
    
    # Replace the anomalous values with nan
    application_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    application_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    
    # Label Encoding pour les colonnes avec <= 2 classes
    uniqueclass_df = application_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)
    list_cols_inf_2_class = uniqueclass_df[uniqueclass_df <= 2].index.tolist()
    
    le = LabelEncoder()
    for col in list_cols_inf_2_class:
        application_train[col] = le.fit_transform(application_train[col])
    
    # One-hot encoding pour les colonnes avec > 2 classes
    app_train_ohe = pd.get_dummies(application_train)
    
    # Même traitement pour test
    uniquetestclass_df = application_test.select_dtypes('object').apply(pd.Series.nunique, axis=0)
    list_cols_inf_2_test_class = uniquetestclass_df[uniquetestclass_df <= 2].index.tolist()
    
    le = LabelEncoder()
    for col in list_cols_inf_2_test_class:
        application_test[col] = le.fit_transform(application_test[col])
    
    app_test_ohe = pd.get_dummies(application_test)
    
    # Sauvegarder les labels et aligner les dataframes
    train_labels = app_train_ohe['TARGET']
    app_train_ohe = app_train_ohe.drop(columns=['TARGET'])
    
    app_train_ohe, app_test_ohe = app_train_ohe.align(app_test_ohe, join='inner', axis=1)
    
    print(f'Training Features shape: {app_train_ohe.shape}')
    print(f'Testing Features shape: {app_test_ohe.shape}')
    
    # ===== POLYNOMIAL FEATURES =====
    poly_features = application_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
    poly_features_test = application_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    
    # Imputer for handling missing values
    imputer = SimpleImputer(strategy='median')
    
    poly_target = poly_features['TARGET']
    poly_features = poly_features.drop(columns=['TARGET'])
    
    # Impute missing values
    poly_features = imputer.fit_transform(poly_features)
    poly_features_test = imputer.transform(poly_features_test)
    
    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree=3)
    
    # Train the polynomial features
    poly_transformer.fit(poly_features)
    
    # Transform the features
    poly_features = poly_transformer.transform(poly_features)
    poly_features_test = poly_transformer.transform(poly_features_test)
    
    # Create a dataframe of the features
    poly_features = pd.DataFrame(poly_features, 
                                  columns=poly_transformer.get_feature_names_out(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                 'EXT_SOURCE_3', 'DAYS_BIRTH']))
    
    # Add in the target
    poly_features['TARGET'] = poly_target
    
    # Put test features into dataframe
    poly_features_test = pd.DataFrame(poly_features_test, 
                                       columns=poly_transformer.get_feature_names_out(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                                      'EXT_SOURCE_3', 'DAYS_BIRTH']))
    
    # Merge polynomial features into training dataframe
    poly_features['SK_ID_CURR'] = application_train['SK_ID_CURR']
    app_train_poly = application_train.merge(poly_features, on='SK_ID_CURR', how='left')
    
    # Merge polynomial features into testing dataframe
    poly_features_test['SK_ID_CURR'] = application_test['SK_ID_CURR']
    app_test_poly = application_test.merge(poly_features_test, on='SK_ID_CURR', how='left')
    
    # Align the dataframes
    app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join='inner', axis=1)
    
    # ===== FEATURES METIER =====
    app_train_domain = application_train.copy()
    app_test_domain = application_test.copy()
    
    app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
    app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
    app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
    app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']
    
    app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
    app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
    app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
    app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']
    
    # ===== SCALING AND FINAL PREPROCESSING =====
    train = app_train_ohe.copy()
    test = app_test_ohe.copy()
    
    # Feature names
    features = list(train.columns)
    
    # Median imputation of missing values
    imputer = SimpleImputer(strategy='median')
    
    # Scale each feature to 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit on the training data
    imputer.fit(train)
    
    # Transform both training and testing data
    train = imputer.transform(train)
    test = imputer.transform(test)
    
    # Repeat with the scaler
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    
    print("✓ Preprocessing terminé")
    
    return train, test, train_labels, features, application_test['SK_ID_CURR']


# ===============================================================
# Fonction de prédiction
# ===============================================================
def make_predictions(model, X_test, threshold=0.4779):
    """
    Fait des prédictions avec le seuil optimal défini dans ton notebook.
    """
    try:
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = np.array([1 if p >= threshold else 0 for p in probabilities])

        print(f"✓ Prédictions effectuées pour {len(predictions)} clients")
        print(f"  - Clients à risque: {predictions.sum()} ({predictions.sum()/len(predictions)*100:.2f}%)")
        print(f"  - Clients sans risque: {(1-predictions).sum()} ({(1-predictions).sum()/len(predictions)*100:.2f}%)")

        return probabilities, predictions
    except Exception as e:
        print(f"✗ Erreur lors des prédictions: {e}")
        raise


# ===============================================================
# Initialisation de l'API
# ===============================================================
@app.on_event("startup")
async def startup_event():
    global model, preprocessed_test_data, test_ids, raw_application_train, raw_application_test

    print("=" * 50)
    print("INITIALISATION DE L'API")
    print("=" * 50)

    # Charger le modèle
    model = load_model()

    # Charger les données
    application_train, application_test = load_data()
    
    # Conserver les données brutes pour les infos clients
    raw_application_train = application_train.copy()
    raw_application_test = application_test.copy()
    
    # Préprocessing
    X_train, X_test, y_train, features, sk_id_curr = preprocess_data(application_train, application_test)

    preprocessed_test_data = X_test
    test_ids = sk_id_curr.values

    print("=" * 50)
    print("API PRÊTE")
    print("=" * 50)


# ===============================================================
# ENDPOINTS
# ===============================================================
@app.get("/")
async def root():
    return {
        "message": "API de Classification de Risque de Crédit",
        "version": "1.0",
        "endpoints": {
            "/predict/{sk_id_curr}": "Prédiction pour un client spécifique",
            "/predict_all": "Prédictions pour tous les clients",
            "/client/{sk_id_curr}": "Informations détaillées d'un client",
            "/client/{sk_id_curr}/summary": "Résumé des informations d'un client",
            "/infos_generales": "Informations générales sur tous les clients",
            "/health": "Vérification de l'état de l'API"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": preprocessed_test_data is not None
    }


@app.get("/predict/{sk_id_curr}")
async def predict_client(sk_id_curr: int):
    try:
        client_idx = np.where(test_ids == sk_id_curr)[0]
        if len(client_idx) == 0:
            raise HTTPException(status_code=404, detail=f"Client {sk_id_curr} non trouvé")

        client_idx = client_idx[0]
        client_data = preprocessed_test_data[client_idx:client_idx+1]
        probability = model.predict_proba(client_data)[0, 1]
        prediction = 1 if probability >= optimal_threshold else 0

        return PredictionResponse(
            sk_id_curr=int(sk_id_curr),
            probability=float(probability),
            prediction=prediction
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict_all")
async def predict_all_clients():
    try:
        probabilities, predictions = make_predictions(model, preprocessed_test_data, optimal_threshold)

        results = []
        for sk_id, prob, pred in zip(test_ids, probabilities, predictions):
            results.append({
                "sk_id_curr": int(sk_id),
                "probability": float(prob),
                "prediction": int(pred)
            })

        return {
            "total_clients": len(results),
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/infos_generales")
async def get_general_info():
    """
    Retourne les informations générales sur l'ensemble des clients de test
    """
    try:
        if raw_application_test is None:
            raise HTTPException(status_code=500, detail="Données non chargées")
        
        nb_credits, rev_moy, credits_moy = load_infos_gen(raw_application_test)
        
        return GeneralInfoResponse(
            nb_credits=nb_credits,
            revenu_moyen=rev_moy,
            credit_moyen=credits_moy
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/client/{sk_id_curr}")
async def get_client_info(sk_id_curr: int):
    """
    Retourne toutes les informations détaillées d'un client
    """
    try:
        if raw_application_test is None:
            raise HTTPException(status_code=500, detail="Données non chargées")
        
        client_data = identite_client(raw_application_test, sk_id_curr)
        
        if client_data is None:
            raise HTTPException(status_code=404, detail=f"Client {sk_id_curr} non trouvé")
        
        return ClientInfoResponse(
            sk_id_curr=sk_id_curr,
            client_data=client_data
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/client/{sk_id_curr}/summary")
async def get_client_summary_endpoint(sk_id_curr: int):
    """
    Retourne un résumé des informations clés d'un client
    """
    try:
        if raw_application_test is None:
            raise HTTPException(status_code=500, detail="Données non chargées")
        
        summary = get_client_summary(raw_application_test, sk_id_curr)
        
        if summary is None:
            raise HTTPException(status_code=404, detail=f"Client {sk_id_curr} non trouvé")
        
        return summary
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===============================================================
# Script principal (exécution directe)
# ===============================================================
if __name__ == "__main__":
    import uvicorn

    print("Démarrage du script de prédiction...")

    # Utiliser les fonctions modifiées
    model = load_model()
    application_train, application_test = load_data()
    
    # Conserver les données brutes
    raw_application_train = application_train.copy()
    raw_application_test = application_test.copy()
    
    X_train, X_test, y_train, features, sk_id_curr = preprocess_data(application_train, application_test)

    probabilities, predictions = make_predictions(model, X_test, optimal_threshold)

    results_df = pd.DataFrame({
        'SK_ID_CURR': sk_id_curr.values,
        'PROBABILITY': probabilities,
        'PREDICTION': predictions
    })

    print("\nDémarrage de l'API FastAPI...")
    uvicorn.run(app, host="0.0.0.0", port=8000)