import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
import os
import time

from config.system_prompt import get_system_prompt
from tools.recherche_juridique import rechercher, formater_contexte


load_dotenv()

# ─────────────────────────────────────────────
# LLM — chargé une seule fois pour toute l'app
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Connexion à Mistral AI...")
def charger_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.3,
        api_key=os.getenv("MISTRAL_API_KEY"),
        streaming=True,
    )

llm = charger_llm()

# ─────────────────────────────────────────────
# Tool de recherche juridique
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Chargement du tool de recherche...")
def charger_tool():
    @tool
    def recherche_juridique(question: str) -> str:
        """Recherche dans la base de données juridique pour répondre à des questions légales."""
        chunks = rechercher(question)
        contexte = formater_contexte(chunks)
        st.session_state.sources_actuelles = chunks
        return contexte

    return recherche_juridique

tool_recherche = charger_tool()

# ─────────────────────────────────────────────
# Agent LangChain (nouvelle API LangGraph)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Configuration de l'agent...")
def charger_agent():
    system_content = get_system_prompt().content
    return create_agent(llm, [tool_recherche], system_prompt=system_content)

agent = charger_agent()

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "historique_messages" not in st.session_state:
    st.session_state.historique_messages = []  # liste de HumanMessage / AIMessage

if "historique" not in st.session_state:
    st.session_state.historique = []  # pour l'affichage

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
    <div class="sous-titre">Votre assistant juridique</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Affichage de l'historique
# ─────────────────────────────────────────────
for message in st.session_state.historique:
    with st.chat_message(message["role"]):
        st.markdown(message["contenu"])

# ─────────────────────────────────────────────
# Saisie utilisateur
# ─────────────────────────────────────────────
question = st.chat_input("Écrivez votre question juridique...")

if question:
    # Afficher + sauvegarder la question
    st.session_state.historique.append({"role": "user", "contenu": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Ajouter au contexte LangChain
    st.session_state.historique_messages.append(HumanMessage(content=question))

    # ── Streaming de la réponse avec l'agent ──────────────────
    debut = time.time()
    with st.chat_message("assistant"):
        placeholder = st.empty()
        reponse_complete = ""

        # Reset sources
        st.session_state.sources_actuelles = []

        # Streamer via la nouvelle API LangGraph (stream_mode="messages")
        for chunk, metadata in agent.stream(
            {"messages": st.session_state.historique_messages},
            stream_mode="messages",
        ):
            # On ne garde que les tokens texte du LLM (noeud "agent")
            if (
                isinstance(chunk, AIMessageChunk)
                and isinstance(chunk.content, str)
                and chunk.content
            ):
                reponse_complete += chunk.content
                placeholder.markdown(reponse_complete + "▌")

        placeholder.markdown(reponse_complete)

    fin = time.time()
    temps_echange = fin - debut

    # Sauvegarder la réponse
    st.session_state.historique_messages.append(AIMessage(content=reponse_complete))
    st.session_state.historique.append({"role": "assistant", "contenu": reponse_complete})
    st.session_state.temps_responses.append(temps_echange)

    # ── Afficher les sources utilisées ──────────────────
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
