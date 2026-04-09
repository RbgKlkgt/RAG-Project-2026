import subprocess
import sys
import os

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

def main():
    print("\n=== Lex-AI ===")
    print("Taper 1  : Echanger avec l'agent dans le terminal (agent local Ollama)")
    print("Taper 1  : Echanger avec l'agent surinterface web (Streamlit)")
    print("q — Quitter\n")

    choix = input("Votre choix : ").strip().lower()

    if choix == "1":
        subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, "agent.py")])

    elif choix == "2":
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            os.path.join(SCRIPTS_DIR, "app_local.py"),
            "--server.address=0.0.0.0",  # fait le lien entre l'extérieur du container et Streamlit (sans ça, Streamlit est invisible depuis ton navigateur)
            "--server.port=8501",         # fait le lien entre ton navigateur (localhost:8501) et Streamlit dans le container
        ])

    elif choix == "q":
        print("Au revoir !")

    else:
        print("Choix invalide.")
        main()

if __name__ == "__main__":
    main()