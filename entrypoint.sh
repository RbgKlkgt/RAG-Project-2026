#!/bin/bash

# 1. Démarre le service Ollama en arrière-plan
echo "Démarrage d'Ollama..."
ollama serve &

# 2. Attend qu'Ollama soit prêt à recevoir des requêtes
echo "Attente d'Ollama..."
until curl -s http://localhost:11434/api/version > /dev/null 2>&1; do
    sleep 1
done
echo "Ollama est prêt."

# 3. Télécharge le modèle Mistral (seulement si absent)
echo "Vérification du modèle Mistral..."
ollama pull mistral
echo "Modèle prêt."

# 4. Lance le menu principal
python main.py
