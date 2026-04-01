import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# Lazy init — chargé uniquement au premier appel
# ─────────────────────────────────────────────
_client = None
_collection = None
_modele = None

def _init():
    global _client, _collection, _modele
    if _modele is None:
        _client = chromadb.PersistentClient(path="./ma_bdd_juridique")
        _collection = _client.get_collection(name="code_civil")
        _modele = SentenceTransformer('all-MiniLM-L6-v2')


def rechercher(question: str, k: int = 4) -> list[dict]:
    """
    Vectorise la question et retourne les k chunks les plus pertinents.

    Retourne une liste de dicts :
    [
        {"texte": "...", "source": "Code civil", "chunk_index": 42, "score": 0.87},
        ...
    ]
    """
    _init()

    # 1. Vectoriser la question avec le même modèle
    vecteur_question = _modele.encode(question).tolist()

    # 2. Recherche par similarité cosinus dans ChromaDB
    resultats = _collection.query(
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
