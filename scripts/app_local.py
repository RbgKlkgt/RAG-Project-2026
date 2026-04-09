import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from config.system_prompt import get_system_prompt

# ─────────────────────────────────────────────
# Définition des loaders (sans appel immédiat)
# ─────────────────────────────────────────────
@st.cache_resource
def charger_librairies():
    from langchain_ollama import ChatOllama
    from langchain_core.tools import tool
    from tools.recherche_juridique_rag import rechercher, formater_contexte
    return ChatOllama, tool, rechercher, formater_contexte

@st.cache_resource
def charger_tool():
    from langchain_core.tools import tool
    from tools.recherche_juridique_rag import rechercher

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
        from tools.recherche_juridique_rag import formater_contexte as _fmt
        st.session_state.sources_actuelles = chunks
        return _fmt(chunks)

    return recherche_juridique

@st.cache_resource
def charger_llm():
    from langchain_ollama import ChatOllama
    tool_recherche = charger_tool()
    llm_base = ChatOllama(model="mistral", temperature=0.3)
    return llm_base.bind_tools([tool_recherche])

@st.cache_resource
def charger_prompt():
    return get_system_prompt()

@st.cache_resource
def prechauffer_base():
    from tools.recherche_juridique_rag import rechercher
    rechercher("initialisation", "code_civil")

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "historique_messages" not in st.session_state:
    st.session_state.historique_messages = []

if "historique" not in st.session_state:
    st.session_state.historique = []

if "temps_responses" not in st.session_state:
    st.session_state.temps_responses = []

if "sources_actuelles" not in st.session_state:
    st.session_state.sources_actuelles = []

# ─────────────────────────────────────────────
# En-tête
# ─────────────────────────────────────────────
st.markdown("""
    <style>
        .titre {
            font-family: 'Augustus', serif;
            text-align: center;
            margin-top: 5vh;
            font-size: 2.5em;
        }
        .sous-titre {
            font-family: 'Augustus', serif;
            text-align: center;
            font-size: 2em;
            margin-top: 0.5em;
        }
    </style>
    <div class="titre">⚖️ Lex-AI ⚖️</div>
    <div class="sous-titre">Votre assistant juridique (local)</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Chargement en 5 étapes avec indicateur
# ─────────────────────────────────────────────
if "pret" not in st.session_state:
    debut_total = time.time()
    with st.status("Initialisation de Lex-AI (local)...", expanded=True) as status:
        barre = st.progress(0)

        # 1/5 — Librairies
        barre.progress(5, text="Étape 1/5 — Chargement des librairies...")
        st.write("⏳ Chargement des librairies...")
        t = time.time()
        charger_librairies()
        st.write(f"✅ Librairies chargées ({time.time() - t:.1f}s)")

        # 2/5 — Tool RAG
        barre.progress(25, text="Étape 2/5 — Chargement du tool RAG...")
        st.write("⏳ Chargement du tool RAG...")
        t = time.time()
        charger_tool()
        st.write(f"✅ Tool RAG chargé ({time.time() - t:.1f}s)")

        # 3/5 — Modèle Mistral
        barre.progress(45, text="Étape 3/5 — Chargement du modèle Mistral (Ollama)...")
        st.write("⏳ Chargement du modèle Mistral (Ollama)...")
        t = time.time()
        charger_llm()
        st.write(f"✅ Modèle Mistral chargé ({time.time() - t:.1f}s)")

        # 4/5 — Prompt système
        barre.progress(65, text="Étape 4/5 — Chargement du prompt système...")
        st.write("⏳ Chargement du prompt système...")
        t = time.time()
        charger_prompt()
        st.write(f"✅ Prompt système chargé ({time.time() - t:.1f}s)")

        # 5/5 — Base vectorielle
        barre.progress(85, text="Étape 5/5 — Connexion à la base vectorielle...")
        st.write("⏳ Connexion à la base vectorielle...")
        t = time.time()
        prechauffer_base()
        st.write(f"✅ Modèle d'embeddings chargé, base vectorielle connectée ({time.time() - t:.1f}s)")

        barre.progress(100, text="Prêt !")
        total = time.time() - debut_total
        status.update(label=f"Lex-AI est prêt ! (chargement en {total:.1f}s)", state="complete", expanded=False)
    st.session_state.pret = True

llm          = charger_llm()
tool_rag     = charger_tool()
system_msg   = charger_prompt()

# ─────────────────────────────────────────────
# Affichage de l'historique
# ─────────────────────────────────────────────
for message in st.session_state.historique:
    with st.chat_message(message["role"]):
        st.markdown(message["contenu"])

# ─────────────────────────────────────────────
# Saisie
# ─────────────────────────────────────────────
question = st.chat_input("Écrivez votre question juridique...")

if question:
    st.session_state.historique.append({"role": "user", "contenu": question})
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.historique_messages.append(HumanMessage(content=question))
    st.session_state.sources_actuelles = []

    debut = time.time()
    with st.chat_message("assistant"):
        placeholder = st.empty()
        reponse_complete = ""

        try:
            # Reconstruction du contexte complet avec le system prompt
            msgs = [system_msg] + st.session_state.historique_messages

            # Boucle d'appel d'outils (comme agent.py)
            with st.spinner("Lex-AI réfléchit..."):
                while True:
                    response = llm.invoke(msgs)
                    msgs.append(response)

                    if not response.tool_calls:
                        break

                    for tool_call in response.tool_calls:
                        placeholder.caption(f"🔍 Recherche : {tool_call['args']}")
                        resultat = tool_rag.invoke(tool_call)
                        msgs.append(ToolMessage(content=resultat, tool_call_id=tool_call["id"]))

            reponse_complete = response.content
            placeholder.markdown(reponse_complete)

        except Exception as e:
            placeholder.error(f"Erreur : {e}")
            reponse_complete = f"Erreur : {e}"

    fin = time.time()
    temps_echange = fin - debut

    st.session_state.historique_messages.append(AIMessage(content=reponse_complete))
    st.session_state.historique.append({"role": "assistant", "contenu": reponse_complete})
    st.session_state.temps_responses.append(temps_echange)

    if st.session_state.sources_actuelles:
        with st.expander("📚 Sources utilisées"):
            for i, source in enumerate(st.session_state.sources_actuelles, 1):
                st.write(f"**Extrait {i}** — {source['source']} (pertinence: {source['score']})")
                st.write(source['texte'])
                st.markdown("---")

# ─────────────────────────────────────────────
# Statistiques de performance
# ─────────────────────────────────────────────
if st.session_state.temps_responses:
    moyenne = sum(st.session_state.temps_responses) / len(st.session_state.temps_responses)
    st.markdown("---")
    st.caption("📊 Statistiques de performance")
    st.write(
        f"Échanges : {len(st.session_state.temps_responses)} | "
        f"Temps moyen : {moyenne:.2f}s | "
        f"Min : {min(st.session_state.temps_responses):.2f}s | "
        f"Max : {max(st.session_state.temps_responses):.2f}s"
    )
