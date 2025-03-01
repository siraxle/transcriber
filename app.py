from openai import OpenAI
import os
from flask import Flask, request, jsonify, send_file
import subprocess
import re
import io

# Инициализация Flask
app = Flask(__name__)

# API_KEYS from envs
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# LLM URLS
LOCAL_LLM_URL = "http://localhost:1234/v1"  # Локальный сервер для LLM
NVIDIA_LLM_URL = "https://integrate.api.nvidia.com/v1"

# Инициализация клиентов OpenAI
local_client = OpenAI(api_key="YOUR_API_KEY", base_url=LOCAL_LLM_URL)  # Для локального сервера
nvidia_client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_LLM_URL)  # Для NVIDIA API

# Папка для сохранения загруженных файлов
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Готовые промпты
PROMPTS = {
    "key_decisions": '''
    Ты помощник по анализу встреч. Выдели ключевые решения и задачи, которые были приняты или поставлены на встрече.
    Выведи их в формате списка, используя Markdown. Не добавляй лишних деталей.
    ''',
    "sentiment_analysis": '''
    Ты помощник по анализу тональности речи. Проанализируй текст и определи, является ли тон речи позитивным, негативным или нейтральным.
    Выведи результат в формате: "Тональность: [позитивная/негативная/нейтральная]". Обоснуй свой вывод.
    ''',
    "generate_questions": '''
    Ты помощник по генерации вопросов. На основе текста встречи сгенерируй список вопросов, которые могут возникнуть у участников.
    Выведи вопросы в формате списка, используя Markdown.
    ''',
    "trend_analysis": '''
    Ты помощник по анализу трендов. Выяви основные тренды или повторяющиеся темы в тексте встречи.
    Выведи их в формате списка, используя Markdown. Например: "Тренд: [описание тренда]".
    ''',
    "summarization": '''
    Ты помощник по суммаризации текстов. Извлекай только самые важные моменты из текста: 
    ключевые идеи, факты, выводы. Не добавляй лишних деталей или интерпретаций. Выводи информацию 
    в формате markdown, придерживаясь четкости и лаконичности. Используй только предоставленный текст и 
    не добавляй информацию о процессе работы модели. Ничего не придумывай, выводи только конспект.
    ''',
}


# Функция для очистки Markdown
def clean_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Удаляем **жирный текст**
    text = re.sub(r'\*(.*?)\*', r'\1', text)  # Удаляем *курсив*
    text = re.sub(r'#+\s*', '', text)  # Удаляем заголовки (#, ##)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Удаляем ссылки [текст](url)
    text = re.sub(r'`(.*?)`', r'\1', text)  # Удаляем `код`
    text = re.sub(r'\n\s*\n', '\n', text)  # Удаляем лишние пустые строки
    return text.strip()


# Главная страница с формой
@app.route('/')
def index():
    return open('templates/index.html', encoding='utf-8').read()


# Эндпоинт для загрузки файла
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    use_local_whisper = request.form.get('use_local_whisper') == 'true'
    llm_model = request.form.get('llm_model')
    prompt_type = request.form.get('prompt_type')  # Тип промпта (готовый или пользовательский)
    custom_prompt = request.form.get('custom_prompt', '')  # Пользовательский промпт

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    print(f"File saved to: {file_path}")

    file_extension = file.filename.split('.')[-1].lower()

    if file_extension in ['txt']:
        transcript = process_text_file(file_path)
    else:
        transcript = transcribe_audio(file_path, use_local_whisper)

    print(f"Transcript: {transcript[:200]}")

    # Выбор промпта
    if prompt_type == "custom":
        prompt = custom_prompt
    else:
        prompt = PROMPTS.get(prompt_type, PROMPTS["key_decisions"])  # По умолчанию "key_decisions"

    response_text, result_filename = process_with_llm(transcript, llm_model, prompt)

    print(f"LLM response: {response_text}")

    return jsonify({
        "transcript": transcript,
        "llm_response": response_text,  # Оригинальный Markdown
        "download_filename": result_filename  # Имя файла для скачивания
    })


# Эндпоинт для скачивания файла
@app.route('/download/<format>/<filename>', methods=['GET'])
def download_file(format, filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if format == 'txt':
        # Чистый текст (без Markdown)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        clean_content = clean_markdown(content)
        return send_file(
            io.BytesIO(clean_content.encode('utf-8')),
            mimetype='text/plain',
            as_attachment=True,
            download_name=f"{filename}.txt"
        )
    elif format == 'md':
        # Оригинальный Markdown
        return send_file(
            file_path,
            mimetype='text/markdown',
            as_attachment=True,
            download_name=f"{filename}.md"
        )
    else:
        return jsonify({"error": "Invalid format"}), 400


# Обработка текстового файла
def process_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().replace('\n', ' ').replace('\r', '')
            return text.strip()
    except Exception as e:
        return f"Error reading text file: {str(e)}"


# Транскрибация аудио
def transcribe_audio(file_path, use_local_whisper):
    if use_local_whisper:
        print('Local Whisper processing...')
        result = subprocess.run(['whisper', file_path], capture_output=True, text=True)
        print(f"Whisper output: {result.stdout}")
        print(f"Whisper error: {result.stderr}")
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error during transcription: {result.stderr.strip()}"
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


# Разбиение текста на части
def split_text(text, max_length=3000):
    parts = []
    while len(text) > max_length:
        split_point = text.rfind(' ', 0, max_length)
        if split_point == -1:
            split_point = max_length
        parts.append(text[:split_point])
        text = text[split_point:].strip()
    if text:
        parts.append(text)
    return parts


# Обработка текста с помощью LLM
def process_with_llm(text, llm_model, prompt):
    print(f"Sending text to LLM: {text[:200]}...")

    text_parts = split_text(text)
    responses = []
    for part in text_parts:
        try:
            response = llm_request(part, llm_model, prompt)
            responses.append(response.choices[0].message.content)
        except Exception as e:
            print(f"Error contacting LLM: {str(e)}")
            responses.append(f"Error contacting LLM: {str(e)}")

    result_text = "\n\n".join(responses)

    # Сохраняем результат в файл
    result_filename = f"result_{llm_model}.md"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(result_text)

    return result_text, result_filename


# Запрос к LLM
def llm_request(text, llm_model, prompt):
    if llm_model == 'local':
        print("Local LLM processing...")
        response = local_client.chat.completions.create(
            model="meta-llama-3.1-8b-instruct",
            messages=[{
                "role": "system",
                "content": prompt
            },
                {
                    "role": "user",
                    "content": text
                }],
            temperature=0.5,
            max_tokens=102400
        )
        return response
    else:
        print(f'NVidia LLM {llm_model} processing...')
        response = nvidia_client.chat.completions.create(
            model=llm_model,
            messages=[{
                "role": "system",
                "content": prompt
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