import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

_VECTOR_STORE = os.getenv("VECTOR_STORE_PATH")
if not _VECTOR_STORE:
    raise EnvironmentError("La variable VECTOR_STORE_PATH n'est pas définie dans .env")

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _VECTOR_STORE)

COLLECTIONS_DISPONIBLES = {
    "code_civil": "Code_civil_1_373",
    "code_impots": "Code_general_des_impots_1_373",
}

# ─────────────────────────────────────────────
# Lazy init — chargé uniquement au premier appel
# ─────────────────────────────────────────────
_client = None
_collections: dict = {}
_modele = None

def _init():
    global _client, _collections, _modele
    if _modele is None:
        _client = chromadb.PersistentClient(path=_DB_PATH)
        _collections = {
            key: _client.get_collection(name=name)
            for key, name in COLLECTIONS_DISPONIBLES.items()
        }
        _modele = SentenceTransformer('all-MiniLM-L6-v2')


def rechercher(question: str, collection: str, k: int = 4) -> list[dict]:
    """
    Vectorise la question et retourne les k chunks les plus pertinents.

    collection : "code_civil" ou "code_impots"

    Retourne une liste de dicts :
    [
        {"texte": "...", "source": "Code civil", "chunk_index": 42, "score": 0.87},
        ...
    ]
    """
    _init()

    if collection not in _collections:
        raise ValueError(f"Collection inconnue : '{collection}'. Valeurs acceptées : {list(_collections.keys())}")

    # 1. Vectoriser la question avec le même modèle
    vecteur_question = _modele.encode(question).tolist()

    # 2. Recherche par similarité cosinus dans ChromaDB
    resultats = _collections[collection].query(
        query_embeddings=[vecteur_question],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    # 3. Formater les résultats
    chunks_pertinents = []
    for i in range(len(resultats["documents"][0])):
        chunks_pertinents.append({
            "texte": resultats["documents"][0][i],
            "source": resultats["metadatas"][0][i].get("source", "inconnu"),
            "chunk_index": resultats["metadatas"][0][i].get("chunk_index", i),
            "score": round(1 - resultats["distances"][0][i], 3)  # distance → similarité
        })

    return chunks_pertinents


def formater_contexte(chunks: list[dict]) -> str:
    """
    Formate les chunks en contexte lisible pour le prompt Mistral.
    """
    contexte = ""
    for i, chunk in enumerate(chunks, 1):
        contexte += f"[Extrait {i} — {chunk['source']} — pertinence : {chunk['score']}]\n"
        contexte += chunk["texte"] + "\n\n"
    return contexte.strip()
