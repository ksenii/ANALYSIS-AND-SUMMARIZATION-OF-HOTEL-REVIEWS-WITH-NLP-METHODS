"""
Preprocessing pipeline for hotel reviews:
- language detection
- HTML removal and text cleaning
- emoji-to-text conversion
- number-to-word transformation
- segmentation with sentence splitting
- stopword filtering
- tokenization
- spell checking

Input: raw review file csv
Output saved in csv-file:
  - original review
  - normalized text
  - lemmatized text
"""

import re
import logging
import sys
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from langdetect import detect
import pymorphy3
from nltk.corpus import stopwords
from num2words import num2words
from razdel import tokenize as razdel_tokenize, sentenize
from spellchecker import SpellChecker
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('text_processing.log', encoding='utf-8'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Initialize analyzer and dictionaries
try:
    morph = pymorphy3.MorphAnalyzer()
    russian_stopwords = set(stopwords.words('russian')).union({"очень", "спасибо", "всё", "это", "который"})
    emoji_translation = {
        "😁": "замечательно", "😅": "замечательно", "😉": "замечательно",
        "😊": "замечательно", "😍": "замечательно", "🔥": "замечательно",
        "❤️": "замечательно", "😒": "ужасно", "🤢": "ужасно", "🙏": "спасибо",
        "✨": "блеск", "⭐️": "звезда", "👍": "класс", "👌🏽": "класс", "👎": "ужасно"}
    @lru_cache(maxsize=10000)
    def cached_lemmatize(word):
        """Cached of lemmatization"""
        return morph.parse(word)[0].normal_form

    spell = SpellChecker(language='ru')
    @lru_cache(maxsize=10000)
    def cached_spell_correction(word):
        """Cached of spell correction"""
        return spell.correction(word)

    CYRILLIC_LETTERS = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
    logger.info("Text processing modules initialized")
except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise


def replace_numbers_with_words(text):
    """Convert numbers to word equivalents."""
    try:
        def replace_match(match):
            num_str = match.group(0)
            try:
                if '.' in num_str:
                    int_part, frac_part = num_str.split('.')
                    int_word = num2words(int(int_part), lang='ru')
                    frac_word = num2words(int(frac_part), lang='ru')
                    return f"{int_word} целых {frac_word} десятых"
                return num2words(int(num_str), lang='ru')
            except:
                return num_str
        text = re.sub(r'(\d+)-ти\b', lambda m: num2words(int(m.group(1)), lang='ru', to='ordinal') + "ти", text)
        text = re.sub(r'(\d+)-х\b', lambda m: num2words(int(m.group(1)), lang='ru', to='ordinal') + "х", text)
        return re.sub(r'\b\d+\.?\d*\b', replace_match, text)
    except Exception as e:
        logger.warning(f"Number conversion error: {str(e)}")
        return text


def replace_emojis(text):
    """Replace emojis with text."""
    try:
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
            "\U00002702-\U000027B0\U000024C2-\U0001F251]", flags=re.UNICODE)

        def repl(match):
            emoji = match.group(0)
            return f" {emoji_translation.get(emoji, emoji)} "
        return emoji_pattern.sub(repl, text)
    except:
        return text


def is_russian_word(word):
    """Check if word contains only Russian letters."""
    try:
        return bool(re.fullmatch(r'^[а-яё]+$', word.lower()))
    except:
        return False


def is_russian_text(text, threshold=0.7):
    """Detect if text is in Russian language."""
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        cyrillic_ratio = sum(1 for char in text if char.lower() in CYRILLIC_LETTERS) / len(text)
        lang = detect(text)
        return cyrillic_ratio >= threshold and lang == 'ru'
    except:
        return False


def clean_text(text):
    """Clean text from HTML, emojis, numbers and special characters."""
    if not isinstance(text, str):
        return ""
    try:
        text = BeautifulSoup(text, "html.parser").get_text()
        text = replace_emojis(text)
        text = replace_numbers_with_words(text)
        text = text.lower()
        text = re.sub(r'[^а-яё.,!?;:\s+-]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        logger.error(f"Text cleaning failed: {str(e)}")
        return ""


def correct_spelling(text):
    """Correct spelling mistakes using cache."""
    if not text.strip():
        return text
    try:
        words = [t.text for t in razdel_tokenize(text)]
        corrected_words = []
        for word in words:
            if word.isdigit() or not is_russian_word(word):
                corrected_words.append(word)
                continue
            correction = cached_spell_correction(word)
            if correction and is_russian_word(correction):
                corrected_words.append(correction)
            else:
                corrected_words.append(word)
        return ' '.join(corrected_words)
    except Exception as e:
        logger.warning(f"Spell checking failed: {str(e)}")
        return text


def segment_text(text):
    """Hybrid text segmentation (rule-based + razdel)"""
    sentences = []
    if not text.strip():
        return sentences
    try:
        text = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
        parts = []
        for part in re.split(r'\s*[-•]\s+', text):
            if part.strip():
                parts.append(part.strip())
        for part in parts:
            for sent in sentenize(part):
                sentence = sent.text
                if not sentence.strip():
                    continue
                if ',' in sentence and len(sentence.split()) > 8:
                    comma_parts = re.split(r'\s*,\s*(?!\s*(?:и|или|но|а)\s*)', sentence)
                    for p in comma_parts:
                        if len(p.split()) >= 2 and is_russian_text(p):
                            sentences.append(p)
                else:
                    if len(sentence.split()) >= 2 and is_russian_text(sentence):
                        sentences.append(sentence)
        return sentences
    except Exception as e:
        logger.error(f"Text segmentation error: {str(e)}")
        return []


def tokenize_and_filter(sentence):
    """Tokenize text and filter short/non-Russian words."""
    try:
        return [t.text for t in razdel_tokenize(sentence)
                if len(t.text) > 2 and is_russian_word(t.text)]
    except Exception as e:
        logger.warning(f"Tokenization failed: {str(e)}")
        return []


def lemmatize_words(words):
    """Lemmatize words and remove stopwords using cache."""
    try:
        lemmas = []
        for word in words:
            if is_russian_word(word):
                lemma = cached_lemmatize(word)
                if lemma not in russian_stopwords:
                    lemmas.append(lemma)
        return lemmas
    except Exception as e:
        logger.warning(f"Lemmatization failed: {str(e)}")
        return []


def process_text(text):
    """Text processing pipeline."""
    result = {'normalized': [], 'lemmatized': []}
    if not text or not isinstance(text, str):
        logger.debug("Empty or invalid text input")
        return result
    try:
        cleaned = clean_text(text)
        if not cleaned:
            return result
        corrected_text = correct_spelling(cleaned)
        sentences = segment_text(corrected_text)
        for s in sentences:
            tokens = tokenize_and_filter(s)
            if len(tokens) < 2:
                continue
            lemmas = lemmatize_words(tokens)
            if len(lemmas) >= 2:
                result['normalized'].append(" ".join(tokens))
                result['lemmatized'].append(" ".join(lemmas))
        return result
    except Exception as e:
        logger.error(f"Text processing failed: {str(e)}")
        return result


def preprocess_reviews(input_file, output_file):
    """Process csv file with hotel reviews."""
    try:
        logger.info("Starting review processing")
        df = pd.read_csv(input_file)
        # Filter Russian reviews
        tqdm.pandas(desc="Filtering Russian reviews")
        df['is_russian'] = df['review'].progress_apply(is_russian_text)
        df = df[df['is_russian']].copy()
        # Process texts
        tqdm.pandas(desc="Processing texts")
        df['processed'] = df['review'].progress_apply(process_text)
        # Extract results
        df['text_normalized'] = df['processed'].apply(lambda x: x['normalized'])
        df['text_lemmatized'] = df['processed'].apply(lambda x: x['lemmatized'])
        df = df[(df['text_normalized'].str.len() > 0) &
                (df['text_lemmatized'].str.len() > 0)]
        # Save results
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return df
    except Exception as e:
        logger.critical(f"Review processing failed: {str(e)}")
        raise
