import os
import numpy as np
import voyageai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from data.queries import DOCUMENTS

load_dotenv()

QDRANT_URL      = "http://localhost:6333"
COL_TRADICIONAL = "tradicional"
COL_FUNNEL      = "funnel_head"
DIM_SHORT       = 128
DIM_FULL        = 1024
REPLICAS        = 500   # 20 docs x Replicas que tu elijas ;), esto con el fin de simular vectores
BATCH           = 500


def embed(texts: list[str]) -> np.ndarray:
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    result = client.embed(texts, model="voyage-4-lite", output_dimension=DIM_FULL)
    return np.array(result.embeddings, dtype=np.float32)


def expandir(embeddings: np.ndarray) -> tuple[np.ndarray, list[str]]:
    # Replica vectores con ruido gaussiano pequeño para que no sean copias exactas
    rng = np.random.default_rng(42)
    bloques_emb  = [embeddings]
    bloques_docs = list(DOCUMENTS)
    for _ in range(REPLICAS - 1):
        ruido = rng.normal(0, 0.01, embeddings.shape).astype(np.float32)
        bloques_emb.append(embeddings + ruido)
        bloques_docs.extend(DOCUMENTS)
    return np.vstack(bloques_emb), bloques_docs


def crear_coleccion(qdrant: QdrantClient, nombre: str, dims: int):
    if qdrant.collection_exists(nombre):
        qdrant.delete_collection(nombre)
    qdrant.create_collection(
        collection_name=nombre,
        vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
    )


def cargar_tradicional(qdrant: QdrantClient, embeddings: np.ndarray, docs: list[str]):
    crear_coleccion(qdrant, COL_TRADICIONAL, DIM_FULL)
    puntos = [
        PointStruct(id=i, vector=emb.tolist(), payload={"texto": doc})
        for i, (doc, emb) in enumerate(zip(docs, embeddings))
    ]
    for i in range(0, len(puntos), BATCH):
        qdrant.upsert(collection_name=COL_TRADICIONAL, points=puntos[i:i+BATCH])
        print(f"  tradicional: {min(i+BATCH, len(puntos))}/{len(puntos)}", end="\r")
    print(f"\n[tradicional]  {len(puntos)} docs | {DIM_FULL} dims en indice")


def cargar_funnel(qdrant: QdrantClient, embeddings: np.ndarray, docs: list[str]):
    crear_coleccion(qdrant, COL_FUNNEL, DIM_SHORT)
    puntos = [
        PointStruct(
            id=i,
            vector=emb[:DIM_SHORT].tolist(),
            payload={"texto": doc, "cola": emb[DIM_SHORT:].tolist()},
        )
        for i, (doc, emb) in enumerate(zip(docs, embeddings))
    ]
    for i in range(0, len(puntos), BATCH):
        qdrant.upsert(collection_name=COL_FUNNEL, points=puntos[i:i+BATCH])
        print(f"  funnel_head: {min(i+BATCH, len(puntos))}/{len(puntos)}", end="\r")
    print(f"\n[funnel_head]  {len(puntos)} docs | {DIM_SHORT} dims en indice, cola en payload")


if __name__ == "__main__":
    qdrant = QdrantClient(url=QDRANT_URL)

    print(f"Generando embeddings para {len(DOCUMENTS)} documentos base")
    embeddings_base = embed(DOCUMENTS)

    print(f"Expandiendo corpus x{REPLICAS} -> {len(DOCUMENTS) * REPLICAS:,} documentos totales")
    embeddings, docs = expandir(embeddings_base)

    cargar_tradicional(qdrant, embeddings, docs)
    cargar_funnel(qdrant, embeddings, docs)

    print(f"\nListo. {len(docs):,} documentos en cada coleccion.")
    print(f"  '{COL_TRADICIONAL}' -> {DIM_FULL} dims en indice")
    print(f"  '{COL_FUNNEL}'      -> {DIM_SHORT} dims en indice + cola en payload")
