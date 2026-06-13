import os
import numpy as np
import voyageai
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from data.queries import QUERIES

load_dotenv()

QDRANT_URL   = "http://localhost:6333"
COL_FUNNEL   = "funnel_head"
DIM_SHORT    = 128
DIM_FULL     = 1024
CANDIDATES_K = 10
TOP_N        = 3


def embed(texts: list[str]) -> np.ndarray:
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    result = client.embed(texts, model="voyage-4-lite", output_dimension=DIM_FULL)
    return np.array(result.embeddings, dtype=np.float32)


def funnel_search(qdrant: QdrantClient, query_emb: np.ndarray) -> list[dict]:
    q_cabeza = query_emb[:DIM_SHORT]
    q_full   = query_emb[:DIM_FULL]

    # Etapa 1: ANN rapido con la cabeza (128 dims)
    candidatos_raw = qdrant.query_points(
        collection_name=COL_FUNNEL,
        query=q_cabeza.tolist(),
        limit=CANDIDATES_K,
        with_payload=True,
        with_vectors=True,
    ).points

    # Etapa 2: rerank reconstruyendo vector completo (cabeza + cola del payload)
    candidatos = []
    for hit in candidatos_raw:
        vector_full = np.concatenate([
            np.array(hit.vector, dtype=np.float32),
            np.array(hit.payload["cola"], dtype=np.float32),
        ])
        candidatos.append({
            "texto":      hit.payload["texto"],
            "score_ann":  hit.score,
            "score_full": float(np.dot(q_full, vector_full)),
        })

    candidatos.sort(key=lambda x: x["score_full"], reverse=True)
    return candidatos[:TOP_N]


if __name__ == "__main__":
    qdrant = QdrantClient(url=QDRANT_URL)

    print(f"Funnel Search — indice: {DIM_SHORT} dims | rerank: {DIM_FULL} dims\n{'='*60}")

    for query in QUERIES:
        print(f"\nQuery: {query}")
        q_emb   = embed([query])[0]
        results = funnel_search(qdrant, q_emb)

        for rank, r in enumerate(results, 1):
            print(
                f"  #{rank} [ann={r['score_ann']:.4f} -> full={r['score_full']:.4f}] "
                f"{r['texto'][:90]}..."
            )
