FROM python:3.12.7-slim

# 1. Installer dépendances système
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       nano unzip curl gcc libpq-dev python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Préparer l'utilisateur (Hugging Face impose l'UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# 3. Installer les dépendances Python
# On le fait avant de copier le reste du code pour profiter du cache Docker
COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir scikit-learn xgboost==2.0.3

# 4. Copier le reste du code
COPY --chown=user . $HOME/app

# 5. CONFIGURATION SPÉCIFIQUE HUGGING FACE / STREAMLIT
EXPOSE 7860

# Configurer Streamlit pour qu'il s'exécute correctement dans le container
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Configurer Streamlit via variables d'environnement (plus propre)
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Lancer l'application avec les bonnes options
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

