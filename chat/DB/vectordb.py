import faiss
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from embedded.embedding import generate_embedding 

class VectorDB:
    def __init__(self, dimension=384, index_file="vector_index.faiss"):
        self.dimension = dimension
        self.index_file = index_file
        self.index = faiss.IndexFlatL2(self.dimension)
        self._load_index()
        self.texts = []

    def _load_index(self):
        
        try:
            self.index = faiss.read_index(self.index_file)
            print(f"Loaded existing index from {self.index_file}")
        except Exception as e:
            print(f"No existing index found. Creating a new one. Error: {e}")

    def _save_index(self):
        
        faiss.write_index(self.index, self.index_file)
        print(f"Index saved to {self.index_file}")

    def add_text(self, text, metadata=None):
        
        embedding = generate_embedding(text)
        embedding = np.array(embedding, dtype=np.float32)
        self.index.add(np.array([embedding], dtype=np.float32))
        print(f"Text added to VectorDB: {text}")

    def search(self, query, top_k=5):
        
        query_embedding = generate_embedding(query)
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), top_k)
        return distances, indices
    
    def get_text_by_index(self, index):
        return self.texts[index]

    def close(self):
        
        self._save_index()
