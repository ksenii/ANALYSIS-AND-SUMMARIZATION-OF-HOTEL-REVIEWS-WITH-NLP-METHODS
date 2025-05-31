"""
Sentiment analysis of hotel reviews
Combines lexicon-based + BERT model for sentiment classification.
Output saved in csv-file
"""

import pandas as pd
import json
import logging
import re
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('sentiment_analysis.log', encoding='utf-8'),
              logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

model_name = 'blanchefort/rubert-base-cased-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


POSITIVE_PATTERNS = {
    'восхитительн*', 'восторг*', 'радост*', 'счастлив*', 'обожа*', 'отличн*', 'любим*', 'наслажд*', 'удовольств*', 'благодарн*', 'супер*',
    'превосходн*', 'идеальн*', 'безупречн*', 'совершенн*', 'отличн*', 'великолепн*', 'потрясающ*', 'фантастич*', 'изумительн*', 'бесподобн*',
    'уютн*', 'комфортн*', 'тепл*', 'гостеприимн*', 'радушн*', "удобн*", 'спокойн*', 'безопасн*', 'приватн*', 'гармоничн*', 'расслаб*', "топов*",
    'внимательн*', 'заботлив*', 'вежлив*', 'доброжелательн*', 'отзывчив*', 'профессионал*', 'компетентн*', 'любезн*', 'предупредительн*',
    'чист*', 'опрятн*', 'аккуратн*', 'свеж*', 'гигиеничн*', 'прекрасн*', "недалек*", 'вкусн*', 'аппетитн*', 'сочн*', 'нежн*', 'ароматн*',
    'современ*', "рядом", "приятн*", 'шедевр*', 'эксклюзив*', 'роскош*', 'премиум*', 'элитн*', "больш*", "качествен*", "клиентоорентир*",
    'ярк*', 'позитив*', 'оптимистич*', 'лёгк*', 'светл*', 'солнечн*', 'дружелюбн*', 'креативн*', 'инновацион*', 'уникальн*', 'стильн*', 'модн*',
    'элегантн*', 'шикарн*', 'блестящ*', 'надёжн*', 'проверен*', 'стабильн*', 'успешн*', 'эффективн*', 'полезн*', 'оздоровительн*', 'релакс*',
    'щедр*', 'душевн*', 'открыт*', 'честн*', 'умн*', 'интересн*', 'увлекательн*', 'занимательн*', 'захватыва*', 'незабываем*',
    'волшебн*', 'сказочн*', 'мечта*', 'ценн*','выгодн*', 'экономичн*', 'доступн*', 'бюджетн*', 'практичн*', 'интуитивн*', 'понятн*',
    'просто*', 'логичн*', 'быстр*', 'оперативн*', 'пунктуальн*', 'дисциплинир*', 'организован*', 'ответствен*',
    'натуральн*', 'свеже*', 'праздничн*', 'торжествен*', 'подарочн*', 'комплимент*', 'поощрительн*', 'бонус*', 'тренд*', 'популярн*', 'востребован*',
    'рекоменд*', 'одобрен*', 'проверен*', 'лучш*', 'высококлассн*', 'престижн*', 'эталон*', 'образцов*', 'иде*', 'мечтательн*',
    'умиротвор*', 'домашн*', 'просторн*', 'стильн*', 'оперативн*','панорам*', 'близост*', 'тих*'
}

NEGATIVE_PATTERNS = {
    'отвратительн*', 'мерзк*', 'ужасн*', 'кошмарн*', 'отвращен*', 'ненавист*', 'раздража*', 'зло*', 'ярост*', 'возмущен*',
    'ужас*', 'катастроф*', 'провал*', 'разочарован*', 'недоволь*', 'неприятн*', 'противн*', 'груб*', 'хамск*', 'хамоват*',
    'гряз*', 'вон*', 'сырост*', 'плесен*', 'запах*', 'тесн*', 'душн*', 'шумн*', 'холодн*', "неудобн*",
    'невнимательн*', 'равнодушн*', 'грубост*', 'халатн*', 'непрофессионал*', 'обман*', 'развод*', 'мошенни*', 'хитрост*', 'нечестн*', "пахнувш*",
    'слом*', 'неисправн*', 'брак*', 'дефект*', 'полом*', 'рухнувш*', 'недочет*', 'ошибк*', 'проблем*', 'недодел*', "рухнув*",
    'антисанитар*', 'зараз*', 'опасн*', 'травмоопас*', 'небольш*', 'отврат*', 'мерзост*', 'гадост*', 'тошнот*', 'блевот*', 'маленьк*', "плох*",
    'безнадёж*', 'бесполезн*', 'бестолков*', 'болезнен*', 'бредов*', 'вредн*', 'глуп*', 'депрессивн*', 'жесток*', 'заброшен*',
    'надоедлив*', 'неадекватн*', 'невыносим*', 'негодн*', 'некорректн*', 'нелеп*', 'неприемлем*', 'неуважительн*', 'отвраща*', 'омерзительн*',
    'оскорбительн*', 'печальн*', 'подл*', 'раздражительн*', 'скучн*', 'страшн*', 'тосклив*',
    'тяжел*', 'худш*', 'чудовищн*', 'шокиру*', 'эксплуататор*', 'ядовит*', 'неприязн*', 'бесит*', 'дешёв*', 'жал*', 'заблуждени*',
    'издеватель*', 'испорчен*', 'нагл*', 'недовери*', 'неудач*', 'отстой*', 'переплат*', 'развалива*',  'халтур*', 'чрезмерн*',
    'дебильн*', 'идиотск*', 'кончен*', 'лаж*', 'несправедлив*', 'перегружен*', 'треш*', 'ущербн*', 'фигов*',
    'невкус*', 'тухл*', 'просроч*', 'фастфуд*', 'маленьк*', 'антисанитар*', 'таракан*',
    'хамств*', 'безразлич*', 'мошенничеств*', 'аварийн*', 'ветх*', 'далек*', 'проходн* двор*', 'промышлен* зон*'
}


def predict_sentiment(text):
    """Predict sentiment using BERT model"""
    if not isinstance(text, str) or not text.strip():
        return 'neutral'
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        label = torch.argmax(probs, dim=1).item()
        return {0: "negative", 1: "neutral", 2: "positive"}[label]
    except Exception as e:
        logger.error(f"BERT sentiment failed: {e}")
        return 'neutral'


def analyze_sentiment(text):
    """Hybrid sentiment analysis lexicon-based + BERT"""
    if not isinstance(text, str):
        return 'neutral'
    words = set(text.lower().split())
    pos = 0
    neg = 0
    for word in words:
        for pattern in POSITIVE_PATTERNS:
            if re.match(pattern.replace('*', '.*'), word):
                pos += 1
        for pattern in NEGATIVE_PATTERNS:
            if re.match(pattern.replace('*', '.*'), word):
                neg += 1
    # Lexicon-based
    if pos > neg:
        return 'positive'
    elif neg > pos:
        return 'negative'
    # BERT if lexicon inconclusive
    return predict_sentiment(text)


def process_theme_fragments(theme_fragments):
    """Add sentiment analysis to theme fragments"""
    if isinstance(theme_fragments, str):
        try:
            themes = json.loads(theme_fragments)
        except json.JSONDecodeError:
            themes = {}
    elif isinstance(theme_fragments, dict):
        themes = theme_fragments
    else:
        themes = {}
    enhanced_themes = defaultdict(list)
    for theme, fragments in themes.items():
        if not isinstance(fragments, list):
            continue
        for fragment in fragments:
            if not isinstance(fragment, str):
                continue
            sentiment = analyze_sentiment(fragment)
            enhanced_themes[theme].append({'fragment': fragment, 'sentiment': sentiment})
    return json.dumps(enhanced_themes, ensure_ascii=False)


def analyze_sentiments(input_file, output_file):
    """Pipeline for sentiment analysis of review"""
    df = pd.read_csv(input_file)
    required_cols = ['theme_fragments']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column {col} missing")
    logger.info(f"Processing {len(df)} reviews")
    df['enhanced_theme_fragments'] = df['theme_fragments'].apply(process_theme_fragments)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")
    return df
