import sys
import os
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

import time

# ─────────────────────────────────────────────
# Chargement avec indicateur de progression
# ─────────────────────────────────────────────
print("\n=== Initialisation de Lex-AI (local) ===")
debut_total = time.time()

print("⏳ [1/5] Chargement des librairies...")
t = time.time()
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
# from langchain_core.messages import  AIMessage
from config.system_prompt import get_system_prompt
from tools.recherche_juridique_rag import rechercher, formater_contexte
print(f"✅ [1/5] Librairies chargées ({time.time() - t:.1f}s)")

print("⏳ [2/5] Chargement du tool RAG...")
t = time.time()

@tool
def recherche_juridique(question: str, collection: str) -> str:
    """Recherche dans la base documentaire juridique RAG.

    Paramètres :
    - question : la question juridique à rechercher
    - collection : la base à interroger.
      Valeurs acceptées :
        "code_civil"    → droit civil (contrats, obligations, famille, successions, propriété)
        "code_impots"   → fiscalité (IR, TVA, IS, plus-values, exonérations fiscales)
    """
    chunks = rechercher(question, collection)
    return formater_contexte(chunks)

print(f"✅ [2/5] Tool RAG chargé ({time.time() - t:.1f}s)")

print("⏳ [3/5] Chargement du modèle Mistral (Ollama)...")
t = time.time()
llm_base = ChatOllama(model="mistral", temperature=0.3)
llm = llm_base.bind_tools([recherche_juridique])
print(f"✅ [3/5] Modèle Mistral chargé ({time.time() - t:.1f}s)")

print("⏳ [4/5] Chargement du prompt système...")
t = time.time()
system_message = get_system_prompt()
print(f"✅ [4/5] Prompt système chargé ({time.time() - t:.1f}s)")

print("⏳ [5/5] Connexion à la base vectorielle...")
t = time.time()
rechercher("initialisation", "code_civil")
print(f"✅ [5/5] Modèle d'embeddings chargé, base vectorielle connectée ({time.time() - t:.1f}s)")

print(f"\nLex-AI est prêt ! (chargement total : {time.time() - debut_total:.1f}s)")

# ─────────────────────────────────────────────
# Boucle de conversation
# ─────────────────────────────────────────────
print("\n=== Chat avec Mistral (local) ===")
print("  • 'exit'           → quitter\n")

historique = [system_message]
temps_responses = []

while True:
    user_input = input("Toi : ").strip()

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

    historique.append(HumanMessage(content=user_input))

    debut = time.time()

    # Boucle d'appel d'outils
    while True:
        response = llm.invoke(historique)
        historique.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            print(f"  [outil] {tool_call['name']}({tool_call['args']})")
            resultat = recherche_juridique.invoke(tool_call)
            historique.append(ToolMessage(content=resultat, tool_call_id=tool_call["id"]))

    fin = time.time()
    temps_echange = fin - debut
    temps_responses.append(temps_echange)

    moyenne_courante = sum(temps_responses) / len(temps_responses)

    print(f"\nMistral : {response.content}")
    print(f"⏱️  Réponse en {temps_echange:.2f}s | Moyenne session : {moyenne_courante:.2f}s\n")
