# 1. Image de base Python légère
FROM python:3.11-slim

# 2. Outils système nécessaires pour installer Ollama
RUN apt-get update && apt-get install -y \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN pip install --no-cache-dir uv

# 3. Installation d'Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# 4. Dossier de travail dans le container
WORKDIR /app

# 5. Installation des dépendances Python
COPY pyproject.toml .
# Install CPU-only torch first to avoid downloading the huge CUDA version
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --system --no-cache .

# 6. Copie du code de l'app
COPY . .

# 7. Rendre le script de démarrage exécutable (obligatoire sur Linux)
RUN chmod +x entrypoint.sh

# 8. Port Streamlit (navigateur → container)
EXPOSE 8501

# 9. Script exécuté au docker run
ENTRYPOINT ["./entrypoint.sh"]
