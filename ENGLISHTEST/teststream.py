import os
import pdfplumber
from sentence_transformers import SentenceTransformer
from groq import Groq
import google.generativeai as genai
import openai
import numpy as np
import os
import sys
import faiss
import streamlit as st
from dotenv import load_dotenv

load_dotenv("C:\\chatbot\\ENGLISHTEST\\.env")
if not os.getenv('GROQ_API_KEY'):
    print("警告: GROQ_API_KEY 未設置")
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

class VectorDB:
    def __init__(self, dimension=384, index_file="vector_index.faiss"):
        self.dimension = dimension
        self.index_file = index_file
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = [] 
        self._load_index()

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
            

    def _save_index(self):
        faiss.write_index(self.index, self.index_file)
        # print(f"Index saved to {self.index_file}")

    def add_text(self, text, metadata=None):
        embedding = generate_embedding(text)  
        embedding = np.array(embedding, dtype=np.float32)
        self.index.add(np.array([embedding], dtype=np.float32))  
        self.texts.append(text)  
        # print(f"Text added to VectorDB: {text}")

    def search(self, query, top_k=5):
        query_embedding = generate_embedding(query)
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), top_k)
        return distances, indices
    
    def get_text_by_index(self, index):
        return self.texts[index]

    def close(self):
        self._save_index()



vector_db = VectorDB()


# 初始化模型
embed_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def generate_embedding(text):
    return embed_model.encode(text)

def chunk_text(text, max_length=512):
    """
    將長文本拆分成較小的段落或句子
    """
    sentences = text.split('.') 
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()  # 去掉多餘的空白
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

def process_text_and_generate_embeddings(text):
    """
    處理長文本
    """
    chunks = chunk_text(text) 
    embeddings = []

    for chunk in chunks:
        embedding = generate_embedding(chunk)
        embeddings.append(embedding)
        # print(f"生成的嵌入: {embedding}")
    return embeddings

sys_msg = (
    "你是一個英文考試(多益)的出題官"
    "嚴格要求不能有非英文的文字或符號出現"
    "如果使用者要求講解或解答時，請完整解釋題目"
    "每次都要完整呈現，不能有缺少"
)

convo = [{'role': 'system', 'content': sys_msg}]

generation_config = {
    'temperature': 0.95,
    'top_p': 1,
    'top_k': 5,
    'max_output_tokens': 2048
}

safety_settings = [
    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'},
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config,
                              safety_settings=safety_settings)

def groq_prompt(prompt):
    convo.append({'role': 'user', 'content': prompt})
    try:
        chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
        response = chat_completion.choices[0].message
        convo.append(response)
        return response.content
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "系統出現錯誤"



def read_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() 
    return text

def read_pdfs_in_folder(folder_path):
    pdf_texts = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                print(f"正在讀取：{pdf_path}")
                text = read_pdf(pdf_path)
                pdf_texts.append(text)
    return pdf_texts

def save_in_txt(texts, output_txt_path):
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for i, text in enumerate(texts, 1):
            f.write(f"--- PDF {i} ---\n")
            f.write(text)
            f.write("\n\n") 
    # print(f"所有文字已儲存至 {output_txt_path}")



def get_relevant_text_from_db(query, top_k=3):

    distances, indices = vector_db.search(query, top_k=top_k)
    print(f"Indices from search: {indices}")
    relevant_texts = vector_db.get_text_by_index(0)
    
    return relevant_texts

def generate_response_with_context(query):

    relevant_texts = get_relevant_text_from_db(query)
    
  
    context = "\n".join(relevant_texts)  
    prompt = f"以下是參考信息，請根據這些信息，並结合你的知識回答問題：\n{context}\n\n問題：{query}\n回答："
    
    response = groq_prompt(prompt)
    return response


st.title("多益題目生成器")

query = st.text_input("請輸入查詢:")

if query:
    folder_path = "C:\\chatbot\\result"
    output_txt_path = "C:\\chatbot\\result.txt"
    pdf_texts = read_pdfs_in_folder(folder_path)
    save_in_txt(pdf_texts, output_txt_path)

    vector_db = VectorDB()
    for text in pdf_texts:
        vector_db.add_text(text)

    response = generate_response_with_context(query)

    st.subheader("生成的回應：")
    st.write(response)