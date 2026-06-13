from qdrant_client import QdrantClient

QDRANT_URL      = "http://localhost:6333"
COLECCIONES     = ["tradicional", "funnel_head"]

if __name__ == "__main__":
    qdrant = QdrantClient(url=QDRANT_URL)
    for col in COLECCIONES:
        if qdrant.collection_exists(col):
            qdrant.delete_collection(col)
            print(f"Eliminada: {col}")
        else:
            print(f"No existe: {col}")
