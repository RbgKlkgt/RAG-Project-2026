import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
import time

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
# Session state
# ─────────────────────────────────────────────
SYSTEM_PROMPT = SystemMessage(content="""
Tu es Lex-AI, un assistant juridique expert en droit français.
Réponds toujours en français, de manière claire et structurée.
Cite les articles de loi pertinents quand c'est possible.
Si tu n'es pas certain d'une information juridique, dis-le explicitement.
""")

if "historique_langchain" not in st.session_state:
    st.session_state.historique_langchain = [SYSTEM_PROMPT]

if "historique" not in st.session_state:
    st.session_state.historique = []

if "temps_responses" not in st.session_state:
    st.session_state.temps_responses = []

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
    st.session_state.historique_langchain.append(HumanMessage(content=question))

    # ── Streaming de la réponse ──────────────────
    debut = time.time()
    with st.chat_message("assistant"):
        placeholder = st.empty()
        reponse_complete = ""

        for chunk in llm.stream(st.session_state.historique_langchain):
            reponse_complete += chunk.content
            placeholder.markdown(reponse_complete + "▌")

        # Remplacer le curseur par le texte final
        placeholder.markdown(reponse_complete)

    fin = time.time()
    temps_echange = fin - debut

    # Sauvegarder la réponse
    st.session_state.historique_langchain.append(AIMessage(content=reponse_complete))
    st.session_state.historique.append({"role": "assistant", "contenu": reponse_complete})
    st.session_state.temps_responses.append(temps_echange)

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