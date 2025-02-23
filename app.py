# TODO возможно import openai не нужно
import openai
from openai import OpenAI

import os
from flask import Flask, request, jsonify
import subprocess

# Инициализация Flask
app = Flask(__name__)

# API_KEYS from envs
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# LLM URLS
LOCAL_LLM_URL = "http://localhost:1234/v1"  # Локальный сервер для LLM
NVIDIA_LLM_URL = "https://integrate.api.nvidia.com/v1"

openai.api_key = "YOUR_API_KEY"  # TODO не нужно?

# Папка для сохранения загруженных файлов
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Главная страница с формой
@app.route('/')
def index():
    return open('templates/index.html').read()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    use_local_whisper = request.form.get('use_local_whisper') == 'true'

    llm_model = request.form.get('llm_model')  # По умолчанию 'local'

    print(f'use_local_whisper={use_local_whisper}, llm_model={llm_model}')

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    print(f"File saved to: {file_path}")  # Логирование для отладки

    # Проверяем тип файла
    file_extension = file.filename.split('.')[-1].lower()

    if file_extension in ['txt']:
        # Обработка текстового файла
        transcript = process_text_file(file_path)
    else:
        # Транскрибируем аудио с помощью Whisper
        transcript = transcribe_audio(file_path, use_local_whisper)

    print(f"Transcript: {transcript[:200]}")  # Логирование транскрипта

    # Отправляем текст в LLM для обработки
    response_text = process_with_llm(transcript, llm_model)

    print(f"LLM response: {response_text}")  # Логирование ответа от LLM

    # Возвращаем результат с правильной кодировкой
    return jsonify({"transcript": transcript, "llm_response": response_text})

def transcribe_audio(file_path, use_local_whisper):
    # Команда для транскрипции с помощью Whisper
    print(type(use_local_whisper))
    if use_local_whisper:
        print('Local Whisper processing...')
        result = subprocess.run(['whisper', file_path], capture_output=True, text=True)

        # Логирование результатов выполнения команды
        print(f"Whisper output: {result.stdout}")  # Покажем stdout
        print(f"Whisper error: {result.stderr}")   # Покажем stderr

        # Если транскрипция прошла успешно, результат будет в stdout
        if result.returncode == 0:
            return result.stdout.strip()  # Возвращаем транскрибированный текст
        else:
            return f"Error during transcription: {result.stderr.strip()}"  # Ошибка транскрипции
    else:
        print('OpenAI API Whisper processing...')
        client = OpenAI(api_key=OPEN_AI_API_KEY)
        audio_file = open(file_path, "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

        print(transcription.text[:200])
        return transcription.text

def process_text_file(file_path):
    # Чтение содержимого текстового файла и удаление лишних символов переноса строки
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Чтение текста и удаление лишних переносов строк
            text = f.read().replace('\n', ' ').replace('\r', '')
            return text.strip()
    except Exception as e:
        return f"Error reading text file: {str(e)}"

def split_text(text, max_length=3000):
    """Разбиение текста на части, чтобы избежать превышения лимита токенов"""
    parts = []
    while len(text) > max_length:
        # Ищем границу (например, на пробелах)
        split_point = text.rfind(' ', 0, max_length)
        if split_point == -1:
            split_point = max_length  # Если нет пробела, разрываем на max_length
        parts.append(text[:split_point])
        text = text[split_point:].strip()
    if text:
        parts.append(text)
    return parts

def process_with_llm(text, llm_model):
    print(f"Sending text to LLM: {text[:200]}...")  # Логируем текст (первые 200 символов для краткости)

    # Разбиваем текст на части, если он слишком длинный
    text_parts = split_text(text)

    responses = []
    for part in text_parts:
        try:
            # Выполняем запрос к API для каждой части текста
            response = llm_request(part, llm_model)

            responses.append(response.choices[0].message.content)

            # TODO поменялось в новой версии библиотеки OpenAI?
            # # Добавляем ответ для каждой части
            # if isinstance(response, ChatCompletion) and 'choices' in response:
            #     if len(response['choices']) > 0:
            #         responses.append(response['choices'][0].get('message', {}).get('content', 'No content returned.'))
            #     else:
            #         responses.append("No valid response from LLM.")
            # else:
            #     responses.append(f"Error: Unexpected response format or missing 'choices' in response: {response}")

        except Exception as e:
            # Логируем ошибку
            print(f"Error contacting LLM: {str(e)}")  # Логирование ошибки
            responses.append(f"Error contacting LLM: {str(e)}")

    # Объединяем ответы из разных частей
    return "\n\n".join(responses)

def llm_request(text, llm_model):
    summary_prompt = '''
    Ты помощник по суммаризации текстов. Извлекай только самые важные моменты из текста: ключевые идеи, факты, выводы. 
    Не добавляй лишних деталей или интерпретаций. Выводи информацию в формате markdown, придерживаясь четкости и лаконичности. 
    Используй только предоставленный текст и не добавляй информацию о процессе работы модели. Ничего не придумывай, выводи только конспект.
    '''

    if llm_model == 'local':
        print("Local LLM processing...")

        # TODO поменялось в новой версии библиотеки OpenAI?
        response = openai.ChatCompletion.create(
            model="meta-llama-3.1-8b-instruct",  # Локальный сервер LLM
            messages=[{
                "role": "system",
                "content": summary_prompt
            },
                {
                    "role": "user",
                    "content": text
                }],
            temperature=0.5,
            max_tokens=102400,
            api_base=LOCAL_LLM_URL  # Локальный сервер LLM
        )
        return response
    else:
        print(f'NVidia LLM {llm_model} processing...')

        client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_LLM_URL)

        response =  client.chat.completions.create(
            model=llm_model,
            messages=[{
                "role": "system",
                "content": summary_prompt
            },
                {
                    "role": "user",
                    "content": text
                }],
            temperature=0.5,
            max_tokens=102400
        )
        return response

if __name__ == '__main__':
    app.run(debug=True)
