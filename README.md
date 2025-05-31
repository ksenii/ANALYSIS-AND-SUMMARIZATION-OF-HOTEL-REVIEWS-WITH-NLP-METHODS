# ANALYSIS-AND-SUMMARIZATION-OF-HOTEL-REVIEWS-WITH-NLP-METHODS
ANALYSIS AND SUMMARIZATION  OF HOTEL REVIEWS WITH NLP METHODS

A web-application for collecting and analyzing hotel reviews from Yandex.Travel platform.

## Key features
- **Review Collection** (download_reviews.py): Collects reviews with date, rating, user authorization status, likes/dislikes
- **Text Preprocessing** (data_preprocessing.py):
  - Language detection
  - HTML removal and text normalization
  - Emoji-to-text conversion
  - Number-to-word transformation
  - Sentence segmentation
  - Stopword removal
  - Lemmatization
  - Spell checking
- **Thematic Analysis** (topic_modeling.py): Categorizes by topics (food, room, cleanliness, service, building, location)
- **Sentiment Analysis**: Categorizes by sentiment (positive, negative or neutral) 
- **Summarization** (sentiment_analysis.py):
  - Extractive (key phrases) (sum_tags.py)
  - Abstractive (using Mistral LLM) (sum_LLM.py)
