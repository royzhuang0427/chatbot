from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def generate_embedding(text):
    
    return model.encode(text)
