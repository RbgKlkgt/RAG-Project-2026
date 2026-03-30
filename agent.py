import time
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader

# --- Chargement du LLM ---
llm = ChatOllama(
    model="mistral",
    temperature=0.3,
)

# --- Chargement du modèle d'embeddings ---
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

print("✅ LLM chargé : Mistral")
print("✅ Embeddings chargés : nomic-embed-text")

# --- Instructions système ---
system_message = SystemMessage(content="""
Tu es un assistant utile et concis.
Réponds toujours en français.
""")

# --- Boucle de conversation ---
print("\n=== Chat avec Mistral (local) ===")
print("  • 'exit'           → quitter\n")

historique = [system_message]
temps_responses = []
document_actif = None      # Texte du PDF chargé
document_nom = None        # Nom du fichier chargé

while True:
    user_input = input("Toi : ").strip()

    # --- Commande : exit ---
    if user_input.lower() == "exit":
        if temps_responses:
            moyenne = sum(temps_responses) / len(temps_responses)
            print(f"\n📊 Récapitulatif de la session :")
            print(f"   • Nombre d'échanges : {len(temps_responses)}")
            print(f"   • Temps moyen       : {moyenne:.2f}s")
            print(f"   • Temps min         : {min(temps_responses):.2f}s")
            print(f"   • Temps max         : {max(temps_responses):.2f}s")
        print("\nAu revoir !")
        break

   
    if not user_input:
        continue

    # --- Envoi du message à Mistral ---
    historique.append(HumanMessage(content=user_input))

    debut = time.time()
    response = llm.invoke(historique)
    fin = time.time()

    temps_echange = fin - debut
    temps_responses.append(temps_echange)

    historique.append(AIMessage(content=response.content))

    moyenne_courante = sum(temps_responses) / len(temps_responses)

    print(f"\nMistral : {response.content}")
    print(f"⏱️  Réponse en {temps_echange:.2f}s | Moyenne session : {moyenne_courante:.2f}s\n")