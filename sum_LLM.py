"""
Abstractive summarization for hotel reviews using LLM
"""

import json
import csv
import logging
from collections import defaultdict, Counter
import ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('sum_LLM.log', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)


def load_reviews_from_csv(file_path):
    """Loads review data from csv file."""
    try:
        reviews = []
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                reviews.append(row)
        return reviews
    except Exception as e:
        logger.error(f"Failed to load csv: {str(e)}")
        raise


def aggregate_fragments(reviews_data):
    """Aggregates fragments by theme and sentiment."""
    aggregated = defaultdict(lambda: defaultdict(list))
    for review in reviews_data:
        if not review.get("enhanced_theme_fragments"):
            continue
        try:
            themes = (json.loads(review["enhanced_theme_fragments"]) if isinstance(review["enhanced_theme_fragments"], str) else review["enhanced_theme_fragments"])
            for theme, fragments in themes.items():
                for fragment_data in fragments:
                    sentiment = fragment_data.get("sentiment", "neutral").lower()
                    fragment = fragment_data["fragment"].strip()
                    if fragment:
                        aggregated[theme][sentiment].append(fragment)
        except Exception as e:
            logger.warning(f"Skipping review error: {str(e)}")
            continue
    for theme in aggregated:
        for sentiment in aggregated[theme]:
            counter = Counter(aggregated[theme][sentiment])
            aggregated[theme][sentiment] = [frag for frag, _ in counter.most_common(20)]
    return aggregated


def build_prompt(aggregated_data):
    """LLM prompt from aggregated data."""
    THEME_ORDER = ['номер', 'еда', 'обслуживание', 'чистота', 'расположение', 'здание']
    sections = []
    for theme in THEME_ORDER:
        theme_data = aggregated_data.get(theme, {})
        positives = theme_data.get('positive', [])
        negatives = theme_data.get('negative', [])
        section = f"\n=== {theme.upper()} ===\n"
        section += "Положительные отзывы:\n" + ("\n".join(f"- {p}" for p in positives) if positives else "Нет положительных отзывов")
        section += "\nОтрицательные отзывы:\n" + ("\n".join(f"- {n}" for n in negatives) if negatives else "Нет отрицательных отзывов")
        sections.append(section)
    return (
        "Ты — эксперт по анализу отзывов об отелях. Вот мнения гостей:\n"
        + "".join(sections) +
        "\n\nСгенерируй аналитический обзор по каждому разделу, соблюдая правила:\n"
        "1. Только факты из отзывов\n"
        "2. Естественный язык\n"
        "3. Начинай с позитива, затем проблемы\n"
        "4. Стиль: как будто делишься личными впечатлениями"
    )


def generate_summaries(input_csv, output_json, models=['mistral']):
    """Function to generate abstractive summaries."""
    try:
        logger.info(f"Starting summarization")
        reviews = load_reviews_from_csv(input_csv)
        aggregated = aggregate_fragments(reviews)
        prompt = build_prompt(aggregated)
        summaries = {}
        for model in models:
            try:
                logger.info(f"Generating with {model}")
                response = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.7})
                summaries[model] = response['message']['content']
                logger.info(f"Successfully generated with {model}")
            except Exception as e:
                logger.error(f"Failed with {model}: {str(e)}")
                summaries[model] = f"Error: {str(e)}"
        output_data = {"prompt": prompt, "summaries": summaries}
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Summarization complete. Results saved")
        return True
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        return False
