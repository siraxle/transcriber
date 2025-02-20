from flask import Flask, request, jsonify
import subprocess
import os
import openai

# Инициализация Flask
app = Flask(__name__)

# Инициализация клиента LLM
openai.api_key = "Whatever"  # Замените на ваш API ключ
base_url = "http://localhost:1234/v1"  # Ваш локальный сервер

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

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Транскрибируем аудио с помощью Whisper (локальный запуск)
    transcript = transcribe_audio(file_path)

    # Отправляем текст в LLM для обработки
    response_text = process_with_llm(transcript)

    # Возвращаем результат с правильной кодировкой
    return jsonify({"transcript": transcript, "llm_response": response_text})

def transcribe_audio(file_path):
    # Команда для транскрибации с помощью Whisper
    result = subprocess.run(['whisper', file_path], capture_output=True, text=True)
    
    # Если транскрипция прошла успешно, результат будет в stdout
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return "Ошибка транскрипции: " + result.stderr.strip()

def process_with_llm(text):
    try:
        response = openai.ChatCompletion.create(
            model="meta-llama-3.1-8b-instruct",  # Модель LLM на локальном сервере
            messages=[
                {"role": "system", "content": "Ты помощник по суммаризации текстов. Выделяй из текста только самое важное. "
                                              "Извлеки информацию из предоставленного текста, не более чем 10 словами."},
                {"role": "user", "content": text}
            ],
            temperature=0.5,
            max_tokens=1024,
            api_base=base_url  # Локальный сервер LLM
        )

        return response.choices[0].message['content']

    except Exception as e:
        return f"Ошибка при обращении к LLM: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
