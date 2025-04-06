import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid

class FAISSVectorStore:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # output size of the embedding model
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_map = {}

    def add_text(self, text, metadata=None):
        embedding = self.model.encode([text])
        vector_id = str(uuid.uuid4())
        self.index.add(np.array(embedding).astype("float32"))
        self.id_map[vector_id] = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {}
        }
        return vector_id

    def search(self, query, top_k=3):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding).astype("float32"), top_k)

        results = []
        for idx in I[0]:
            if idx == -1: continue
            for k, v in self.id_map.items():
                if np.array_equal(v["embedding"], self.index.reconstruct(idx)):
                    results.append({"id": k, "text": v["text"], "metadata": v["metadata"]})
                    break
        return results

faiss_vector_store = FAISSVectorStore()
