from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from PIL import ImageGrab, Image
import speech_recognition as sr
import openai
import google.generativeai as genai
import pyperclip
import cv2
import pyttsx3
import re
import emoji

groq_client = Groq(api_key = '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')
genai.configure(api_key = 'AIzaSyBLixsVDiAbkORF8JV03Af_aNAFy5yeE4s')
openai.api_key = ('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')
web_cam = cv2.VideoCapture(0)
recognizer = sr.Recognizer()

app = Flask(__name__)
CORS(app)

sys_msg = (
    'You are a multi-model AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added '
    'Use all of the context of this conversation so your response os relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity. You only respond in Chinese. Do not switch'
    'to English, regardless of the input language.'
)

convo = [{'role': 'system', 'content': sys_msg}]

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 30,
    'max_output_tokens': 2048
}

safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE'
    },
{
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE'
    },
{
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE'
    },
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config = generation_config,
                              safety_settings = safety_settings
                              )

def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n    IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    try:
        chat_completion = groq_client.chat.completions.create(messages=convo, model = 'llama3-70b-8192')
        response = chat_completion.choices[0].message
        convo.append(response)
        return response.content
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "系統出現錯誤"

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content,'
        'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond'
        'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will'
        'respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"] \n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the'
        'function call name exactly as I listed. You only respond in Chinese. Do not switch to English, regardless of'
        'the input language.'
    )

    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message

    return response.content

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality = 15)

def web_cam_capture():
    if not web_cam.isOpened():
        print('Error: Camera did not open successfully')
        exit()

    path = 'webcam.jpg'

    ret, frame = web_cam.read()
    cv2.imwrite(path, frame)
    web_cam.release()

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('No clipboard text to copy')
        return None

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and try extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    return response.text

def speak(text):
    # 初始化文本到語音引擎
    engine = pyttsx3.init()
    # 設置語速
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate -30)  # 减慢语速
    # 設置音量
    volume = engine.getProperty('volume')
    engine.setProperty('volume', min(1.0, volume))  # 增加音量，最高为 1.0
    # 朗讀讀文本
    engine.say(text)
    engine.runAndWait()

# def voice_input():
#     with sr.Microphone() as source:
#         recognizer.adjust_for_ambient_noise(source) #調整環境噪音
#         recognizer.pause_threshold = 1.5 #減少靜默時間
        #audio = recognizer.listen(source)  # 監聽說話
        # try:
        #     # 使用 Google 的语音识别将音频转换为文本
        #     spoken_text = recognizer.recognize_google(audio, language='zh-CN')
        #     print(f"您說: {spoken_text}")
        #     return spoken_text
        # except sr.UnknownValueError:
        #     print("未能識別，請重試。")
        #     return None
        # except sr.RequestError as e:
        #     print(f"無法連接到語音服務; {e}")
        #     return None

def filter_emojis(response):
    return emoji.replace_emoji(response, replace=' ')

visual_context = None

@app.route('/process', methods=['POST'])
def process():
    user_input = request.json.get('input')

    #初始化visual_context
    global visual_context
    if visual_context is None:
        visual_context = " "

    response = groq_prompt(user_input, img_context = visual_context)
    filtered_response = filter_emojis(response)

    call = function_call(user_input)

    if 'take screenshot' in call:
        print('Taking screenshot')
        take_screenshot()
        visual_context = vision_prompt(prompt=user_input, photo_path='screenshot.jpg')
    elif 'capture webcam' in call:
        print('Capturing webcam')
        web_cam_capture()
        visual_context = vision_prompt(prompt=user_input, photo_path='webcam.jpg')
    elif 'extract clipboard' in call:
        print('Copying clipboard text')
        paste = get_clipboard_text()
        prompt = f'{user_input}\n\n CLIPBOARD CONTEXT: {paste}'
        visual_context = None
    
    #response = groq_prompt(prompt=prompt, img_context=visual_context)
    # max_length = 50
    # current_line = ""
    # for char in response:
    #     current_line += char
    #     if len(current_line) > max_length:
    #         print(current_line)
    #         current_line = ""
    # if current_line:
    #     print(current_line)

    #filtered_response = filter_emojis(response)
    #speak(filtered_response)
    return jsonify({'response': filtered_response})

if __name__ == '__main__':
    app.run(debug = True)





