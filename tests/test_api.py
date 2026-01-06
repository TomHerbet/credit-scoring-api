import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

# Ajout du chemin courant pour permettre l'import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import de votre application
import api_v4

# =============================================================================
# CLASSES ET DONNÉES FACTICES (MOCKS)
# =============================================================================

class MockModel:
    """Simule le modèle LightGBM/Sklearn"""
    def predict_proba(self, X):
        # Retourne des probabilités aléatoires mais cohérentes en dimension
        # X est un array numpy, on retourne [1 - p, p] pour chaque ligne
        n_samples = X.shape[0]
        # On force une probabilité fixe pour tester le résultat déterministe
        # Si X[0,0] > 0.5 (juste pour l'exemple) on met un risque élevé
        probs = np.zeros((n_samples, 2))
        probs[:, 1] = 0.6  # Toujours 60% de risque pour le test
        probs[:, 0] = 0.4
        return probs

@pytest.fixture
def mock_data():
    """Crée des petits DataFrames pandas pour simuler les fichiers CSV"""
    # Création d'un DataFrame de test factice (Raw data)
    df_test = pd.DataFrame({
        'SK_ID_CURR': [100001, 100002, 100003],
        'AMT_INCOME_TOTAL': [50000.0, 100000.0, 75000.0],
        'AMT_CREDIT': [200000.0, 500000.0, 300000.0],
        'AMT_ANNUITY': [10000.0, 25000.0, 15000.0],
        'DAYS_BIRTH': [-10000, -15000, -20000],
        'CODE_GENDER': ['M', 'F', 'M'],
        # Ajout des colonnes minimales pour éviter les erreurs de clés
        'NAME_FAMILY_STATUS': ['Single', 'Married', 'Married'],
        'CNT_CHILDREN': [0, 1, 2],
        'NAME_EDUCATION_TYPE': ['Secondary', 'Higher', 'Higher'],
        'NAME_INCOME_TYPE': ['Working', 'Working', 'Pensioner'],
        'NAME_CONTRACT_TYPE': ['Cash loans', 'Cash loans', 'Revolving loans'],
        'DAYS_EMPLOYED': [-2000, -5000, 365243],
        'OCCUPATION_TYPE': ['Laborers', 'Core staff', None],
        'ORGANIZATION_TYPE': ['Business Entity', 'School', 'XNA']
    })
    
    df_train = df_test.copy() # On utilise le même format pour le train
    
    # Création des données préprocessées (Matrix X)
    # 3 clients, 10 features (nombre arbitraire pour le test)
    X_test = np.random.rand(3, 10)
    
    # Liste des IDs correspondants
    test_ids = pd.Series([100001, 100002, 100003])
    
    features = [f'feature_{i}' for i in range(10)]
    
    return {
        'train': df_train,
        'test': df_test,
        'X_test': X_test,
        'test_ids': test_ids,
        'features': features
    }

# =============================================================================
# FIXTURE CLIENT (SETUP)
# =============================================================================

@pytest.fixture
def client(mock_data):
    """
    Crée un client de test FastAPI en remplaçant les fonctions lourdes
    (load_model, load_data, preprocess) par des Mocks.
    """
    # On "patch" les fonctions de chargement pour ne pas lire les vrais fichiers
    with patch('api_v4.load_model') as mock_load_model, \
         patch('api_v4.load_data') as mock_load_data, \
         patch('api_v4.preprocess_data') as mock_preprocess:
        
        # Configuration des retours des mocks
        mock_load_model.return_value = MockModel()
        mock_load_data.return_value = (mock_data['train'], mock_data['test'])
        
        # preprocess_data retourne: train, test, train_labels, features, sk_id_curr
        # On ne se soucie pas de train/labels ici, on renvoie None pour simplifier
        mock_preprocess.return_value = (
            None, 
            mock_data['X_test'], 
            None, 
            mock_data['features'], 
            mock_data['test_ids']
        )
        
        # IMPORTANT : On doit réinitialiser les variables globales de api_v4
        # car le TestClient va déclencher l'événement 'startup'
        with TestClient(api_v4.app) as test_client:
            yield test_client

# =============================================================================
# TESTS UNITAIRES
# =============================================================================

def test_root(client):
    """Teste la racine de l'API"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "API de Classification de Risque de Crédit"

def test_health(client):
    """Teste l'endpoint de santé"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["data_loaded"] is True

def test_predict_client_valid(client):
    """Teste la prédiction pour un client existant (100001)"""
    sk_id = 100001
    response = client.get(f"/predict/{sk_id}")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["sk_id_curr"] == sk_id
    assert "probability" in data
    assert "prediction" in data
    # Dans notre MockModel, la proba est fixée à 0.6
    assert data["probability"] == 0.6
    # Le seuil est ~0.47, donc prediction devrait être 1
    assert data["prediction"] == 1

def test_predict_client_not_found(client):
    """Teste la prédiction pour un client inexistant"""
    sk_id = 999999
    response = client.get(f"/predict/{sk_id}")
    assert response.status_code == 404
    assert "non trouvé" in response.json()["detail"]

def test_infos_generales(client):
    """Teste les statistiques générales"""
    response = client.get("/infos_generales")
    assert response.status_code == 200
    data = response.json()
    
    # Basé sur nos mock_data :
    # Revenus: 50k, 100k, 75k -> Moyenne 75k
    # Crédits: 200k, 500k, 300k -> Moyenne 333,333.33
    assert data["nb_credits"] == 3
    assert data["revenu_moyen"] == 75000.0
    assert 333000 < data["credit_moyen"] < 334000

def test_client_info_full(client):
    """Teste la récupération complète des infos client"""
    sk_id = 100002
    response = client.get(f"/client/{sk_id}")
    assert response.status_code == 200
    data = response.json()
    
    assert data["sk_id_curr"] == sk_id
    # Vérification d'une valeur spécifique du mock
    assert data["client_data"]["AMT_INCOME_TOTAL"] == 100000.0
    assert data["client_data"]["CODE_GENDER"] == "F"

def test_client_summary(client):
    """Teste le résumé client"""
    sk_id = 100001
    response = client.get(f"/client/{sk_id}/summary")
    assert response.status_code == 200
    data = response.json()
    
    # Vérifie que les champs clés sont présents
    keys_to_check = ["SK_ID_CURR", "AGE", "AMT_INCOME_TOTAL", "AMT_CREDIT"]
    for key in keys_to_check:
        assert key in data
    
    # Vérification du calcul de l'âge (-10000 jours / -365 ~= 27 ans)
    assert data["AGE"] == 27

def test_predict_all(client):
    """Teste l'endpoint de prédiction globale"""
    response = client.get("/predict_all")
    assert response.status_code == 200
    data = response.json()
    
    assert data["total_clients"] == 3
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["sk_id_curr"] == 100001