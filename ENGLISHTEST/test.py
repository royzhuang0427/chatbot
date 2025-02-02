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
    raise ValueError("GROQ_API_KEY 未設置")
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

sys_msg = (
    "You are an experienced English teacher specializing in TOEIC preparation. "
    # "Your responsibilities include: "
    # "1. Creating TOEIC-style questions when requested "
    # "2. Providing detailed explanations for answers "
    # "3. Teaching relevant grammar points and vocabulary "
    # "4. Giving study tips and strategies "
    # "5. Correcting student's English mistakes "
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



folder_path = "C:\\chatbot\\ENGLISHTEST\\source"
output_txt_path = "C:\\chatbot\\ENGLISHTEST\\result.txt"


def get_relevant_text_from_db(query, top_k=3):
    distances, indices = vector_db.search(query, top_k=top_k)
    relevant_texts = [vector_db.get_text_by_index(i) for i in indices]
    combined_text = " ".join(relevant_texts)
    return combined_text[:2000] if combined_text else ""

    
def dynamic_weighting(response):
    score = len(response)
    return score

def generate_response_with_context(query):
    try:
        relevant_texts = get_relevant_text_from_db(query)
        if not relevant_texts:
            print("警告: 未找到相關文本，直接使用查詢生成回應")
            return groq_prompt(query)
        else:
            prompt = (
                f"For non-test questions, provide a normal response without these separators."
                f"As an English teacher, please help with the following request while considering "
                f"this reference material: {relevant_texts}\n\n"
                f"Student's request: {query}\n\n"
                f"If this is a request for multiple questions, please format each question as follows:\n"
                f"===QUESTION START===\n"
                f"[Question and options (A,B,C,D)]\n"
                f"===EXPLANATION===\n"
                f"Correct Answer: [letter]\n"
                f"[Explanation]\n"
                f"===QUESTION END===\n\n"       
            )
            response = groq_prompt(prompt)


        if "===QUESTION START===" in response and "===QUESTION END===" in response:
            questions = response.split("===QUESTION START===")[1:]
            questions = [q.split("===QUESTION END===")[0].strip() for q in questions]
            
            for i, question in enumerate(questions, 1):
                try:

                    if "===EXPLANATION===" in question:
                        question_part, explanation_part = question.split("===EXPLANATION===")
                        question_part = question_part.strip()
                        explanation_part = explanation_part.strip()
                        
                        if "Correct Answer:" in explanation_part:
                            correct_answer = explanation_part.split("Correct Answer:")[1].strip()[0]
                            if correct_answer in ['A', 'B', 'C', 'D']:
                                question_id = len(scoring_system.correct_answers) + 1
                                scoring_system.add_question(question_id, correct_answer, 1.0)
                                
                                print(f"\n第 {i} 題：")
                                print(question_part)
                                
                                user_answered = process_user_answer(question_id, correct_answer)
                              
                                if user_answered:
                                    print("\n本題解析：")
                                    print(explanation_part)
                                    print("\n" + "="*50)
                                

                                if i < len(questions): 
                                    input("\n按 Enter 繼續下一題...")
                                    
                except Exception as e:
                    print(f"處理第 {i} 題時出錯: {e}")
                    continue
                    
        else:
            print(f"\n回應：{response}")
        
        return response
        
    except Exception as e:
        print(f"生成回應時出錯: {e}")
        return groq_prompt(query)

def process_user_answer(question_id, correct_answer):
    while True:
        try:
            user_answer = input(f"\n請輸入您的答案 (A/B/C/D) 問題 {question_id}，或輸入 'skip' 跳過: ").strip().upper()
            if user_answer == 'SKIP':
                return False
            if user_answer in ['A', 'B', 'C', 'D']:
                scoring_system.record_user_answer(question_id, user_answer)
                if user_answer == correct_answer:
                    print("\n正確答案！")
                else:
                    print(f"\n錯誤答案！正確答案是: {correct_answer}")
                return True
            print("無效的輸入，請輸入 A、B、C、D 或 'skip'")
        except Exception as e:
            print(f"處理答案時出錯: {e}")
            return False

def save_response_to_txt(query, response, output_file="responses.txt"):
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# GenAI 回應記錄\n\n")
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"用戶提問: {query}\n") 
        f.write(f"{'='*50}\n")
        f.write(response)
        f.write("\n\n")

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
    try:
        vector_db = VectorDB()
        if not is_db_initialized:
            pdf_texts = vector_db.read_pdfs_in_folder(folder_path)
            if pdf_texts:
                save_source_in_txt(pdf_texts, output_txt_path) 
                for text in pdf_texts:
                    if vector_db.add_text(text):
                        print("添加了新文本")
                    else:
                        print("文本已存在，跳過處理")
                is_db_initialized = True
            else:
                print("警告: 沒有找到PDF文件或文件讀取失敗")

        while True:
            try:
                query = input("請輸入您的問題（輸入'quit'退出，'score'查看成績）: ")
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'score':
                    show_results()
                    continue
                elif query.lower() == 'reembedding':
                    vector_db.reembedding(folder_path)
                    continue
                response = generate_response_with_context(query)
                save_response_to_txt(query, response)
                print("\nGenAI 回覆:", response)
                
                print("\n" + "="*50 + "\n")
            except Exception as e:
                print(f"處理查詢時出錯: {str(e)}")

    except Exception as e:
        print(f"初始化過程出錯: {str(e)}")
    finally:
        vector_db.close()  