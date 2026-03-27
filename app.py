import streamlit as st

# Initialisation de l'historique
if "historique" not in st.session_state:
    st.session_state.historique = []

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
            font-size:2em;
            margin-top: 0.5em;
        }
            
    </style>
    <div class="titre">⚖️ Lex-AI ⚖️</div>
    <div class="sous-titre">Votre assistant juridique</div>
""", unsafe_allow_html=True)

# Afficher l'historique
for message in st.session_state.historique:
    with st.chat_message(message["role"]):
        st.markdown(message["contenu"])

# Zone de saisie
question = st.chat_input("Écrivez votre question juridique...")

if question:
    # Sauvegarder la question
    st.session_state.historique.append({
        "role": "user",
        "contenu": question
    })

    # Afficher la question
    with st.chat_message("user"):
        st.markdown(question)

    # --- ICI on branchera le RAG ---
    reponse = f"Réponse simulée à : {question}"

    # Sauvegarder la réponse
    st.session_state.historique.append({
        "role": "assistant",
        "contenu": reponse
    })

    # Afficher la réponse
    with st.chat_message("assistant"):
        st.markdown(reponse)
