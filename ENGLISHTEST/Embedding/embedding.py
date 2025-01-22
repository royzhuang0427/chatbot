from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    _instance = None
    
    def __new__(cls, model_name='paraphrase-MiniLM-L6-v2'):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance.model = SentenceTransformer(model_name)
        return cls._instance

    def generate_embedding(self, text):
        return self.model.encode(text)
    
    
    @staticmethod
    def chunk_text(text, max_length=512):
        sentences = text.split('.') 
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= max_length:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks
    
    def process_text_and_generate_embeddings(self, text):
        chunks = self.chunk_text(text)
        embeddings = []
        for chunk in chunks:
            embedding = self.generate_embedding(chunk)
            embeddings.append(embedding)
        return embeddings
