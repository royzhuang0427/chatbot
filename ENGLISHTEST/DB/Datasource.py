import os
import numpy as np
import faiss
import pdfplumber
from Embedding.embedding import EmbeddingModel



class VectorDB:
    def __init__(self, dimension=384, index_file="vector_index.faiss"):
        self.dimension = dimension
        self.embedding_model = EmbeddingModel()
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        self.index_file = os.path.join(script_dir, index_file)
        self.text_hash_file = os.path.join(script_dir, "text_hash_file.txt")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.text_hashes = set()
        self._load_index()
        self._load_text_hashes()

    def _load_text_hashes(self):
        if os.path.exists(self.text_hash_file):
            with open(self.text_hash_file, 'r') as f:
                self.text_hashes = set(f.read().splitlines())

    def _save_text_hashes(self):
        with open(self.text_hash_file, 'w') as f:
            f.write('\n'.join(self.text_hashes))

    def _load_index(self):
        if os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file) 
                # print(f"Loaded existing index from {self.index_file}")
            except Exception as e:
                print(f"Error loading index: {e}")
        else:
            print(f"No existing index found. Creating a new one.")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts = [] 

    def add_text(self, text, metadata=None):
        text_hash = str(hash(text))
        if text_hash in self.text_hashes:
            return False
        embedding = self.embedding_model.generate_embedding(text)
        embedding = np.array(embedding, dtype=np.float32)
        self.index.add(np.array([embedding], dtype=np.float32))
        self.texts.append(text)
        self.text_hashes.add(text_hash)
        self._save_text_hashes()
        self._save_index()
        return True

    def search(self, query, top_k=5):
        try:
            query_embedding = self.embedding_model.generate_embedding(query)
            query_embedding = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.index.search(query_embedding, top_k)

            indices = indices[0].tolist()
            return distances[0], indices
        except Exception as e:
            print(f"搜索時出錯: {e}")
            return [], []

    def get_text_by_index(self, index):
        try:
            if isinstance(index, (list, np.ndarray)):
                index = int(index[0])
            else:
                index = int(index)
            return self.texts[index]
        except (IndexError, TypeError, ValueError) as e:
            print(f"獲取文本時出錯: {e}")
            return "未找到相關文本"
        
    def _save_index(self):
        try:
            faiss.write_index(self.index, self.index_file)
            print(f"索引已保存到 {self.index_file}")
        except Exception as e:
            print(f"保存索引時出錯: {e}")    
        
    def clear_all(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.text_hashes = set()
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.text_hash_file):
            os.remove(self.text_hash_file)
        print("已清除所有向量数据")

    def close(self):
        self._save_index()
        self._save_text_hashes()

    
    @staticmethod
    def read_pdf(pdf_path):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"讀取PDF時出錯 {pdf_path}: {str(e)}")
            return ""

    def read_pdfs_in_folder(self, folder_path):
        if not os.path.exists(folder_path):
            print(f"錯誤: 文件夾不存在 {folder_path}")
            return []
        
        all_texts = []
        try:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_path = os.path.join(root, file)
                        print(f"正在讀取PDF：{pdf_path}")
                        text = self.read_pdf(pdf_path)
                        if text:
                            all_texts.append(text)
        except Exception as e:
            print(f"讀取文件夾時出錯: {str(e)}")
        
        return all_texts


    def reembedding(self, folder_path):
        try:
            print("開始重新embedding...")
            self.clear_all()
            
            pdf_texts = self.read_pdfs_in_folder(folder_path)
            if not pdf_texts:
                print("警告: 沒有讀取到任何PDF文本")
                return
            
            for text in pdf_texts:
                if self.add_text(text):
                    print("添加了新文本")
            
            print("重新embedding完成")

        except Exception as e:
            print(f"重新embedding時出錯: {str(e)}")