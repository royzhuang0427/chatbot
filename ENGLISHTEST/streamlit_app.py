import streamlit as st
import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
from DB.Datasource import VectorDB
from utils.scoring_utils import AnswerScoring
from utils.text_utils import save_source_in_txt

SYS_MSG = (
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
vevtor_db = VectorDB()

def init_app():
    st.session_state.setdefault('scoring_system', AnswerScoring())
    st.session_state.setdefault('conversation', [{'role': 'system', 'content': SYS_MSG}])
    st.session_state.setdefault('current_question', None)
    st.session_state.setdefault('vector_db', VectorDB())
    st.session_state.setdefault('is_db_initialized', False)
    st.session_state.setdefault('current_questions', [])
    st.session_state.setdefault('user_answers', {})
    st.session_state.setdefault('question_data', {})
    st.session_state.setdefault('submitted', False)

def init_api_clients():
    load_dotenv("C:\\chatbot\\ENGLISHTEST\\.env")
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        st.warning("警告: GROQ_API_KEY 未設置")
        return None
    return Groq(api_key=api_key)

def get_relevant_text_from_db(query, top_k=3):
    try:
        distances, indices = st.session_state.vector_db.search(query, top_k=top_k)
        relevant_texts = [st.session_state.vector_db.get_text_by_index(i) for i in indices]
        combined_text = " ".join(relevant_texts)
        return combined_text[:2000] if combined_text else ""
    except Exception as e:
        st.error(f"搜索相關文本時出錯: {str(e)}")
        return ""

def call_groq_api(prompt, groq_client):
    st.session_state.conversation.append({'role': 'user', 'content': prompt})
    if len(st.session_state.conversation) > 10:
        st.session_state.conversation = [st.session_state.conversation[0]] + st.session_state.conversation[-9:]

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=st.session_state.conversation,
            model='llama3-70b-8192'
        )
        response = chat_completion.choices[0].message
        st.session_state.conversation.append(response)
        return response.content
    except Exception as e:
        st.error(f"調用Groq API時出錯: {str(e)}")
        st.session_state.conversation.pop()
        return None

def create_prompt(query, relevant_texts):
    return (
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
        f"For non-test questions, provide a normal response without these separators."
    )

def save_response(query, response):
    output_file = "responses.txt"
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        mode = 'a' if output_path.exists() else 'w'
        with open(output_path, mode, encoding='utf-8') as f:
            if mode == 'w':
                f.write("# GenAI 回應記錄\n\n")
            f.write(f"\n{'='*50}\n")
            f.write(f"用戶提問: {query}\n")
            f.write(f"{'='*50}\n")
            f.write(response)
            f.write("\n\n")
    except Exception as e:
        st.error(f"保存回應時出錯: {str(e)}")

def main():
    st.set_page_config(page_title="TOEIC 英語測試助手", layout="wide")
    st.title("TOEIC 英語測試助手")

    init_app()
    groq_client = init_api_clients()
    if not groq_client:
        st.error("無法初始化 API 客戶端，請檢查設置")
        return

    with st.sidebar:
        st.header("功能選單")
        if st.button("查看成績"):
            show_results()
        if st.button("重新加載資料庫"):
            reload_database()
        if st.button("重新嵌入"):
            with st.spinner("正在重新嵌入..."):
                st.session_state.vector_db.reembedding(folder_path = r"C:\chatbot\ENGLISHTEST\result")
                st.success("重新嵌入完成")
        if st.button("重置測驗"):
            st.session_state.user_answers = {}
            st.session_state.current_questions = []
            st.session_state.question_data = {}
            st.session_state.submitted = False
            st.experimental_rerun()

        if st.session_state.current_questions:
            total_questions = len(st.session_state.current_questions)
            answered_questions = len([ans for ans in st.session_state.user_answers.values() if ans in ['A', 'B', 'C', 'D']])
            st.write(f"已作答: {answered_questions}/{total_questions} 題")
            if answered_questions > 0:
                st.write("你的答案:", "".join([st.session_state.user_answers.get(i, '_') for i in range(1, total_questions + 1)]))

    query = st.text_input("請輸入您的問題：")
    if st.button("提交", key="submit_query"):
        if query:
            with st.spinner("正在生成回應..."):
                response = generate_response_with_context(query, groq_client)
                if response:
                    save_response(query, response)
                    display_response(response)


def generate_response_with_context(query, groq_client):
    try:
        relevant_texts = get_relevant_text_from_db(query)
        if not relevant_texts:
            st.warning("警告: 未找到相關文本，直接使用查詢生成回應")

        prompt = create_prompt(query, relevant_texts)
        return call_groq_api(prompt, groq_client)
    except Exception as e:
        st.error(f"生成回應時出錯: {str(e)}")
        return None

def display_response(response):
    if not response:
        return

    if "===QUESTION START===" in response and "===QUESTION END===" in response:
        if not st.session_state.current_questions:
            questions = response.split("===QUESTION START===")[1:]
            questions = [q.split("===QUESTION END===")[0].strip() for q in questions]
            st.session_state.current_questions = questions

            for idx, question in enumerate(questions, 1):
                if "===EXPLANATION===" in question:
                    question_part, explanation_part = question.split("===EXPLANATION===")
                    correct_answer = explanation_part.split("Correct Answer:")[1].strip()[0]
                    st.session_state.question_data[idx] = {
                        'question': question_part.strip(),
                        'explanation': explanation_part.strip(),
                        'correct_answer': correct_answer
                    }

        for idx in range(1, len(st.session_state.current_questions) + 1):
            question_data = st.session_state.question_data[idx]
            with st.expander(f"問題 {idx}"):
                st.markdown(question_data['question'])

        if not st.session_state.submitted:
            col1, col2 = st.columns([3, 1])
            with col1:
                answers = st.text_input("請輸入答案（例如：ABCD）:", key="answers_input")
            with col2:
                submit = st.button("提交答案", key="submit_all")
            
            if submit and answers:
                if not all(ans in ['A', 'B', 'C', 'D'] for ans in answers.upper()):
                    st.error("答案只能包含 A、B、C、D")
                    return
                if len(answers) != len(st.session_state.current_questions):
                    st.error(f"請輸入 {len(st.session_state.current_questions)} 個答案")
                else:
                    correct_count = 0
                    total = len(st.session_state.current_questions)

                    for idx, ans in enumerate(answers.upper(), 1):
                        if ans in ['A', 'B', 'C', 'D']:
                            st.session_state.user_answers[idx] = ans
                            if ans == st.session_state.question_data[idx]['correct_answer']:
                                correct_count += 1

                    st.session_state.submitted = True
                    st.session_state.score = (correct_count / total) * 100

                    st.experimental_rerun()


        else:
            correct_count = 0
            total_questions = len(st.session_state.current_questions)
            for idx in range(1, total_questions + 1):
                question_data = st.session_state.question_data[idx]
                user_answer = st.session_state.user_answers.get(idx)
                correct_answer = question_data['correct_answer']
                is_correct = user_answer == correct_answer
                
                with st.expander(f"問題 {idx} 結果"):
                    st.write(f"你的答案: {user_answer}")
                    st.write(f"正確答案: {correct_answer}")
                    st.write(f"{'✓ 正確!' if is_correct else '✗ 錯誤!'}")
                    st.write("解釋:")
                    st.write(question_data['explanation'] if question_data['explanation'] else "無")
                    if is_correct:
                        correct_count += 1
            
            st.success(f"總分: {st.session_state.score:.1f}% ({correct_count}/{total_questions})")

def show_results():
    if not st.session_state.user_answers:
        st.warning("還沒有任何測驗記錄")
        return
        
    correct_count = 0
    total_questions = len(st.session_state.current_questions)
    
    st.header("測驗結果")
    for idx, answer in st.session_state.user_answers.items():
        correct_answer = st.session_state.question_data[idx]['correct_answer']
        is_correct = answer == correct_answer
        if is_correct:
            correct_count += 1
        
        st.write(f"問題 {idx}: {'✓' if is_correct else '✗'}")
        
    if total_questions > 0:
        score = (correct_count / total_questions) * 100
        st.write(f"總分: {score:.1f}% ({correct_count}/{total_questions})")

def reload_database():
    with st.spinner("正在重新加載資料庫..."):
        st.session_state.vector_db = VectorDB()
        st.session_state.is_db_initialized = True
        st.success("資料庫已重新加載")

if __name__ == "__main__":
    main()
