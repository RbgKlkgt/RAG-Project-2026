from pathlib import Path
from langchain_core.messages import SystemMessage


def get_system_prompt() -> SystemMessage:
    """Lit l'instruction système depuis un fichier (bonne pratique)."""
    fiche = Path(__file__).parent / "system_prompt.txt"
    texte = fiche.read_text(encoding="utf-8").strip()
    return SystemMessage(content=texte)
