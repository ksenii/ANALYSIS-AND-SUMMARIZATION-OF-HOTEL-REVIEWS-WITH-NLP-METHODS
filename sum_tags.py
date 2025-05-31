"""
Sentiment analysis and thematic clustering of hotel reviews
Output saved in csv-file:
- theme
- sentiment
- cluster metrics
- representative phrases
- cluster percentage
"""

import json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import hdbscan
import pymorphy3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('sum_tags.log', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
morph = pymorphy3.MorphAnalyzer()


def phrase_is_informative(phrase: str) -> bool:
    """Filters out low-value phrases using linguistic rules."""
    words = phrase.strip().split()
    if len(words) < 2:
        return False
    tags = []
    for w in words:
        parses = morph.parse(w)
        if not parses:
            tags.append(None)
            continue
        tags.append(parses[0])
    if tags[0] and tags[0].tag.POS in {'PRCL', 'INTJ', 'CONJ'}:
        if len(tags) > 1 and tags[1] and tags[1].tag.POS == 'NOUN':
            return False
    if len(words) == 2 and words[0].lower() in {'минус', 'нет', 'не'} and tags[1] and tags[1].tag.POS == 'NOUN':
        return False
    if len(tags) == 2 and tags[0] and tags[1]:
        if tags[0].tag.POS == 'ADVB' and tags[1].tag.POS in {'VERB', 'INFN'}:
            return False
    for i in range(len(tags) - 1):
        if tags[i] and tags[i+1]:
            if tags[i].tag.POS == 'NOUN' and tags[i+1].tag.POS == 'NOUN':
                return False
    has_verb = any(t and t.tag.POS in {'VERB', 'INFN'} for t in tags)
    has_adj = any(t and t.tag.POS in {'ADJF', 'ADJS'} for t in tags)
    has_adv = any(t and t.tag.POS == 'ADVB' for t in tags)
    return has_verb or has_adj or has_adv


def correct_adjective_noun_phrase(phrase):
    """Corrects adjective-noun phrases"""
    words = phrase.split()
    if len(words) < 2:
        return phrase
    adj, noun = words[0], words[1]
    try:
        noun_parse = morph.parse(noun)[0]
        if 'NOUN' not in noun_parse.tag:
            return phrase
        noun_gender = noun_parse.tag.gender
        noun_number = noun_parse.tag.number
        noun_case = noun_parse.tag.case or 'nomn'
        adj_parse = morph.parse(adj)[0]
        if noun_gender and noun_number:
            new_adj = adj_parse.inflect({noun_gender, noun_number, noun_case})
            adj_corrected = new_adj.word if new_adj else adj
        else:
            adj_corrected = adj
        return f"{adj_corrected} {noun}" + (" " + " ".join(words[2:]) if len(words) > 2 else "")
    except Exception as e:
        logger.error(f"Grammar correction failed '{phrase}': {str(e)}")
        return phrase


def save_results(results, output_file="clustering_results.csv"):
    """Saves clustering results to csv"""
    csv_data = []
    for result in results:
        for cluster_info in result["cluster_data"]:
            try:
                rep_sentence_corrected = correct_adjective_noun_phrase(cluster_info["representative_sentence"])
                csv_data.append({
                    "theme": result["theme"],
                    "sentiment": result["sentiment"],
                    "silhouette_score": float(result["silhouette_score"]) if result["silhouette_score"] is not None else None,
                    "cluster_id": int(cluster_info["cluster_id"]),
                    "representative_sentence": rep_sentence_corrected,
                    "cluster_percentage": float(cluster_info["cluster_percentage"])})
            except Exception as e:
                logger.error(f"Error processing cluster {cluster_info}: {str(e)}")
                continue
    try:
        pd.DataFrame(csv_data).to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"Results saved: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise


def load_models():
    """Loads embedding model"""
    logger.info("Loading embedding model...")
    try:
        embed_model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded successfully")
        return embed_model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def extract_fragments_by_sentiment(df):
    """
    Extracts text fragments grouped by sentiment.
    Applies weighting based on:
    - Likes/dislikes ratio
    - Authorization status
    - Rating
    """
    pos = defaultdict(list)
    neu = defaultdict(list)
    neg = defaultdict(list)
    for _, row in df.iterrows():
        try:
            theme_data = json.loads(row['enhanced_theme_fragments'].replace("'", '"').replace('""', '"'))
            weight = (row.get('likes', 0) + 1) / (row.get('dislikes', 0) + 1)
            if row.get('authorization', False):
                weight *= 1.2
            try:
                rating = float(row.get('rating', 3))
                weight *= rating / 3
            except (ValueError, TypeError):
                pass
            for theme, items in theme_data.items():
                for item in items:
                    frag = item.get("fragment", "").strip()
                    sentiment = item.get("sentiment", "").lower()
                    if not frag or len(frag.split()) < 2:
                        continue
                    if not phrase_is_informative(frag):
                        continue
                    entry = (frag, float(round(weight, 4)))
                    if sentiment == "positive":
                        pos[theme].append(entry)
                    elif sentiment == "neutral":
                        neu[theme].append(entry)
                    elif sentiment == "negative":
                        neg[theme].append(entry)
        except Exception as e:
            logger.warning(f"Error processing {row.name}: {str(e)}")
            continue
    return pos, neu, neg


def find_representative_sentence(cluster_embeddings, cluster_fragments, cluster_weights):
    """Finds the most representative sentence in a cluster"""
    try:
        similarities = cosine_similarity(cluster_embeddings, cluster_embeddings.mean(axis=0).reshape(1, -1)).flatten()
        idx = np.argmax(similarities * np.array(cluster_weights))
        return cluster_fragments[idx]
    except Exception as e:
        logger.error(f"Error finding representative: {str(e)}")
        return cluster_fragments[0] if cluster_fragments else ""


def cluster_theme_fragments(fragments_with_weights, theme, sentiment, model, total_fragments):
    """Performs clustering on text fragments for theme+sentiment."""
    if len(fragments_with_weights) < 3:
        return {
            "theme": theme,
            "sentiment": sentiment,
            "model": MODEL_NAME,
            "fragments": len(fragments_with_weights),
            "clusters": 0,
            "silhouette_score": None,
            "cluster_data": []}
    try:
        fragments, weights = zip(*fragments_with_weights)
        embeddings = model.encode(fragments, show_progress_bar=False)
        embeddings = normalize(embeddings)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean')
        clusterer.fit(embeddings)
        labels = clusterer.labels_
        if len(set(labels)) <= 1:
            return {
                "theme": theme,
                "sentiment": sentiment,
                "model": MODEL_NAME,
                "fragments": len(fragments),
                "clusters": 1,
                "silhouette_score": 0.0,
                "cluster_data": [{
                    "cluster_id": 0,
                    "representative_sentence": fragments[0],
                    "cluster_percentage": 0}]}
        score = float(silhouette_score(embeddings, labels))
        clusters = defaultdict(lambda: {"fragments": [], "weights": [], "embeddings": []})
        for i, label in enumerate(labels):
            clusters[label]["fragments"].append(fragments[i])
            clusters[label]["weights"].append(weights[i])
            clusters[label]["embeddings"].append(embeddings[i])
        cluster_data = []
        for cluster_id, cluster_info in clusters.items():
            rep_sentence = find_representative_sentence(
                np.vstack(cluster_info["embeddings"]),
                cluster_info["fragments"],
                cluster_info["weights"])
            percent = 100 * len(cluster_info["fragments"]) / total_fragments
            cluster_data.append({
                "cluster_id": cluster_id,
                "representative_sentence": rep_sentence,
                "cluster_percentage": percent})
        return {
            "theme": theme,
            "sentiment": sentiment,
            "model": MODEL_NAME,
            "fragments": len(fragments),
            "clusters": len(clusters),
            "silhouette_score": score,
            "cluster_data": cluster_data
        }
    except Exception as e:
        logger.error(f"Clustering failed for {theme}-{sentiment}: {str(e)}")
        return {
            "theme": theme,
            "sentiment": sentiment,
            "model": MODEL_NAME,
            "fragments": len(fragments_with_weights),
            "clusters": 0,
            "silhouette_score": None,
            "cluster_data": []}


def sum_tags_process(input_csv, output_csv):
    """Extractive pipeline"""
    try:
        df = pd.read_csv(input_csv)
        model = load_models()
        pos, neu, neg = extract_fragments_by_sentiment(df)
        results = []
        for theme in set(list(pos.keys()) + list(neu.keys()) + list(neg.keys())):
            for sentiment, fragments in (("positive", pos.get(theme, [])), ("neutral", neu.get(theme, [])), ("negative", neg.get(theme, []))):
                if not fragments:
                    continue
                result = cluster_theme_fragments(fragments, theme, sentiment, model, total_fragments=len(fragments))
                results.append(result)
        save_results(results, output_csv)
        logger.info("Processing completed successfully.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
