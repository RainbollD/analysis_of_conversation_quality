import spacy
from transformers import pipeline

# Загрузка моделей
nlp = spacy.load("ru_core_news_sm")
sentiment_analyzer = pipeline("sentiment-analysis")


def analyze_call_transcript(transcript):
    # Анализ тональности
    sentiment = sentiment_analyzer(transcript)
    sentiment_score = sentiment[0]['score'] * (1 if sentiment[0]['label'] == 'POSITIVE' else -1)

    # Определение ключевых тем
    doc = nlp(transcript)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Оценка по 10-балльной шкале
    score = (sentiment_score + 1) * 5

    # Рекомендации
    recommendations = []
    if sentiment_score < 0:
        recommendations.append("Улучшить вежливость.")
    if 'complaint' in transcript.lower():
        recommendations.append("Обратить внимание на жалобы.")

    return {
        "sentiment_score": sentiment_score,
        "entities": entities,
        "score": score,
        "recommendations": recommendations
    }


# Пример использования
transcript = "Спасибо. Но могло быть лучше"
result = analyze_call_transcript(transcript)
print(result)
