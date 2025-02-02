import os
import pdfplumber
import numpy as np
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv
from DB.Datasource import VectorDB
from Embedding.embedding import EmbeddingModel
from utils.text_utils import save_source_in_txt
from utils.scoring_utils import AnswerScoring
from pathlib import Path

is_db_initialized = False

scoring_system = AnswerScoring()

load_dotenv("C:\\chatbot\\ENGLISHTEST\\.env")
if not os.getenv('GROQ_API_KEY'):
    print("警告: GROQ_API_KEY 未設置")
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

sys_msg = (
    "You are an experienced English teacher specializing in TOEIC preparation. "
    "Your responsibilities include: "
    "1. Creating TOEIC-style questions when requested "
    "2. Providing detailed explanations for answers "
    "3. Teaching relevant grammar points and vocabulary "
    "4. Giving study tips and strategies "
    "5. Correcting student's English mistakes "
    "Please ensure all responses are in proper English. "
    "When explaining, be thorough but easy to understand. "
    "Always maintain a supportive and encouraging teaching tone."
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
MAX_CONVERSATION_LENGTH = 10

def groq_prompt(prompt):
    if len(convo) > MAX_CONVERSATION_LENGTH:
        convo[:] = [convo[0]] + convo[-MAX_CONVERSATION_LENGTH+1:]

    convo.append({'role': 'user', 'content': prompt})
    try:
        chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
        response = chat_completion.choices[0].message
        convo.append(response)
        return response.content
    except Exception as e:
        print(f"調用Groq API時出錯: {e}")
        convo.pop()
        return f"系統出現錯誤：{str(e)}"

output_txt_path = "C:\\chatbot\\ENGLISHTEST\\result.txt"

def show_results():
    total_score, percentage_score, wrong_questions = scoring_system.calculate_score()
    stats = scoring_system.get_statistics()
    
    print("\n" + "="*50)
    print("測驗結果:")
    print(f"總分: {percentage_score:.2f}%")
    print(f"答題數: {stats['answered_questions']}/{stats['total_questions']}")
    print(f"正確率: {stats['accuracy']:.2f}%")
    
    if wrong_questions:
        print("\n錯誤題目:")
        for wrong in wrong_questions:
            print(f"題號 {wrong['question_id']}:")
            print(f"您的答案: {wrong['user_answer']}")
            print(f"正確答案: {wrong['correct_answer']}\n")
    print("="*50 + "\n")

if __name__ == "__main__":
        while True:
            try:
                query = input("請輸入您的問題（輸入'quit'退出，'score'查看成績）: ")
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'score':
                    show_results()
                    continue
                response = groq_prompt(query)
                print("\nGenAI 回覆:", response)
                
                print("\n" + "="*50 + "\n")
            except Exception as e:
                print(f"處理查詢時出錯: {str(e)}")
