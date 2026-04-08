import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
import os
import time
import httpx

from config.system_prompt import get_system_prompt


load_dotenv()

# ─────────────────────────────────────────────
# Définition des loaders (sans appel immédiat)
# ─────────────────────────────────────────────
@st.cache_resource
def charger_llm():
    return ChatMistralAI(
        model="open-mistral-7b",
        temperature=0.3,
        api_key=os.getenv("MISTRAL_API_KEY"),
        streaming=True,
    )

@st.cache_resource
def charger_tool():
    from tools.recherche_juridique import rechercher, formater_contexte

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
        contexte = formater_contexte(chunks)
        st.session_state.sources_actuelles = chunks
        return contexte

    return recherche_juridique

@st.cache_resource
def charger_agent():
    llm = charger_llm()
    tool_recherche = charger_tool()
    system_content = get_system_prompt().content
    return create_agent(llm, [tool_recherche], system_prompt=system_content)

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

if "quota_depasse" not in st.session_state:
    st.session_state.quota_depasse = False

# ─────────────────────────────────────────────
# En-tête — affiché immédiatement
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
# Bannière d'avertissement quota
# ─────────────────────────────────────────────
if st.session_state.quota_depasse:
    st.error(
        "⚠️ **Quota API Mistral dépassé.** "
        "Les nouvelles questions sont temporairement suspendues. "
        "Veuillez patienter que le quota soit réinitialisé (généralement dans quelques minutes), "
        "puis cliquez sur le bouton ci-dessous pour reprendre."
    )
    if st.button("Réessayer"):
        st.session_state.quota_depasse = False
        st.rerun()

# ─────────────────────────────────────────────
# Chargement différé avec indicateur de progression
# ─────────────────────────────────────────────
if "pret" not in st.session_state:
    debut_total = time.time()
    with st.status("Initialisation de Lex-AI...", expanded=True) as status:
        barre = st.progress(0)

        barre.progress(5, text="Étape 1/4 — Connexion à Mistral AI...")
        st.write("⏳ Connexion à Mistral AI...")
        time.sleep(0.05)
        t = time.time()
        charger_llm()
        st.write(f"✅ Mistral AI connecté ({time.time() - t:.1f}s)")

        barre.progress(25, text="Étape 2/4 — Chargement de l'outil de recherche juridique...")
        st.write("⏳ Chargement de l'outil de recherche juridique...")
        time.sleep(0.05)
        t = time.time()
        charger_tool()
        st.write(f"✅ Outil de recherche juridique chargé ({time.time() - t:.1f}s)")

        barre.progress(50, text="Étape 3/4 — Chargement du modèle d'embeddings et connexion à la base vectorielle...")
        st.write("⏳ Chargement du modèle d'embeddings et connexion à la base vectorielle...")
        time.sleep(0.05)
        t = time.time()
        from tools.recherche_juridique import rechercher as _prewarm
        _prewarm("initialisation")
        st.write(f"✅ Modèle d'embeddings chargé, base vectorielle connectée ({time.time() - t:.1f}s)")

        barre.progress(80, text="Étape 4/4 — Configuration de l'agent...")
        st.write("⏳ Configuration de l'agent...")
        time.sleep(0.05)
        t = time.time()
        charger_agent()
        st.write(f"✅ Agent configuré ({time.time() - t:.1f}s)")

        barre.progress(100, text="Prêt !")
        total = time.time() - debut_total
        status.update(label=f"Lex-AI est prêt ! (chargement en {total:.1f}s)", state="complete", expanded=False)
    st.session_state.pret = True

agent = charger_agent()

# ─────────────────────────────────────────────
# Affichage de l'historique
# ─────────────────────────────────────────────
for message in st.session_state.historique:
    with st.chat_message(message["role"]):
        st.markdown(message["contenu"])

# ─────────────────────────────────────────────
# Saisie
# ─────────────────────────────────────────────
question = st.chat_input(
    "Écrivez votre question juridique...",
    disabled=st.session_state.quota_depasse,
)

if question and not st.session_state.quota_depasse:
    contenu_message = question
    label_historique = question

    st.session_state.historique.append({"role": "user", "contenu": label_historique})
    with st.chat_message("user"):
        st.markdown(label_historique)

    st.session_state.historique_messages.append(HumanMessage(content=contenu_message))

    # ── Streaming de la réponse avec l'agent ──────────────────
    debut = time.time()
    with st.chat_message("assistant"):
        placeholder = st.empty()
        reponse_complete = ""

        st.session_state.sources_actuelles = []

        try:
            for chunk, metadata in agent.stream(
                {"messages": st.session_state.historique_messages},
                stream_mode="messages",
            ):
                if (
                    isinstance(chunk, AIMessageChunk)
                    and isinstance(chunk.content, str)
                    and chunk.content
                ):
                    reponse_complete += chunk.content
                    placeholder.markdown(reponse_complete + "▌")

            placeholder.markdown(reponse_complete)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                st.session_state.quota_depasse = True
                placeholder.warning(
                    "⚠️ Quota API Mistral dépassé. "
                    "Impossible de répondre pour l'instant. "
                    "Le quota se réinitialise généralement en quelques minutes."
                )
                reponse_complete = "⚠️ Quota API dépassé — réponse indisponible."
            else:
                placeholder.error(f"Erreur API ({e.response.status_code}) : {e}")
                reponse_complete = f"Erreur API : {e}"

        except Exception as e:
            placeholder.error(f"Erreur inattendue : {e}")
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

    if st.session_state.quota_depasse:
        st.rerun()

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
