from flask import Flask, render_template, request, jsonify
from download_reviews import extract_reviews
from data_preprocessing import preprocess_reviews
from topic_modeling import analyze_themes
from sentiment_analysis import analyze_sentiments
from sum_tags import sum_tags_process
from sum_LLM import generate_summaries
import os
import logging
import csv
import threading
import json

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('pipeline.log', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

status_messages = []
is_running = False
processing_complete = False
output_filename = "processed_reviews.csv"
themes_filename = "reviews_with_themes.csv"
final_filename = "final_analyzed_reviews.csv"
clustering_output = "clustering_results.csv"
summary_output = "theme_summaries.json"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_scraping', methods=['POST'])
def start_scraping():
    global is_running, status_messages, processing_complete
    if is_running:
        return jsonify({"error": "Процесс уже запущен"}), 400
    url = request.form.get('url')
    months = int(request.form.get('months', 12))
    status_messages = []
    processing_complete = False
    is_running = True
    thread = threading.Thread(target=run_pipeline, args=(url, months))
    thread.start()
    return jsonify({"message": "Обработка началась"})


def run_pipeline(url, months):
    global is_running, status_messages, processing_complete

    def update_status(message):
        status_messages.append(message)
        logger.info(message)
    try:
        update_status("Сбор отзывов...")
        raw_file = "raw_reviews.csv"
        extract_reviews(url=url, output_csv=raw_file, max_months_old=months)
        if not os.path.exists(raw_file):
            update_status("Отзывы не были собраны")
            return
        update_status("Предобработка текста...")
        preprocess_reviews(raw_file, output_filename)
        update_status("Анализ тем...")
        analyze_themes(output_filename, themes_filename)
        update_status("Анализ тональности...")
        analyze_sentiments(themes_filename, final_filename)
        update_status("Экстрактивная суммаризация...")
        sum_tags_process(final_filename, clustering_output)
        update_status("Абстрактивная суммаризация...")
        success = generate_summaries(
            input_csv=final_filename,
            output_json=summary_output,
            models=['mistral'])
        if not success:
            update_status("Ошибка при генерации суммаризации")
        else:
            update_status("Суммаризация завершена")
        update_status("Обработка завершена!")
        processing_complete = True
    except Exception as e:
        logger.error(f"Ошибка пайплайна: {str(e)}")
        update_status(f"Ошибка: {str(e)}")
    finally:
        is_running = False


@app.route('/get_status')
def get_status():
    return jsonify({
        "messages": status_messages,
        "is_running": is_running,
        "processing_complete": processing_complete})


@app.route('/get_clustering_results')
def get_clustering_results():
    results = []
    if os.path.exists(clustering_output):
        with open(clustering_output, encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                try:
                    percentage = float(row['cluster_percentage'].replace(',', '.'))
                except Exception as e:
                    continue
                if percentage == 0.0:
                    continue
                percentage = round(percentage, 1)
                theme_raw = row.get('theme')
                theme_clean = 'неопределено' if not theme_raw else theme_raw.strip()
                sentiment_raw = row.get('sentiment', 'neutral').strip()
                sentence_raw = row.get('representative_sentence', '').strip()
                results.append({
                    'theme': theme_clean,
                    'sentiment': sentiment_raw,
                    'sentence': sentence_raw,
                    'percentage': percentage})
    else:
        logger.error(f"Файл {clustering_output} не найден")
    return jsonify({"results": results})


@app.route('/get_summary')
def get_summary():
    if os.path.exists(summary_output):
        try:
            with open(summary_output, 'r', encoding='utf-8') as f:
                data = json.load(f)
            summaries = data.get("summaries", {})
            return jsonify({
                "summaries": summaries,
                "available_models": list(summaries.keys())}), 200
        except Exception as e:
            logger.error(f"Ошибка чтения JSON: {e}")
            return jsonify({"error": "Ошибка чтения файла суммаризации"}), 500
    else:
        return jsonify({"error": "Суммаризация ещё не готова"}), 404


if __name__ == '__main__':
    app.run(debug=True)
