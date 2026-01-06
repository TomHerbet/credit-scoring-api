FROM python:3.11.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements.txt
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'API
COPY api_v4.py .

# Exposer le port 8000
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "api_v4:app", "--host", "0.0.0.0", "--port", "8000"]