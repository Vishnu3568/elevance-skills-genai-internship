"""Sentiment Analyzer module for Customer Service Chatbot.

Provides sentiment detection (positive, negative, neutral) using the 
cardiffnlp/twitter-roberta-base-sentiment-latest Hugging Face model.
"""

from typing import Dict, Any, Optional
from transformers import pipeline

_classifier: Optional[Any] = None

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def get_sentiment_classifier():
    """Lazily load and return the Hugging Face sentiment analysis pipeline."""
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME
        )
    return _classifier


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze the sentiment of a given input text.

    Args:
        text (str): Input text string to analyze.

    Returns:
        dict: A dictionary containing:
            - 'label': Sentiment label ('positive', 'negative', or 'neutral')
            - 'score': Confidence score (float)
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected input text to be a string, got {type(text).__name__}")

    stripped_text = text.strip()
    if not stripped_text:
        raise ValueError("Input text cannot be empty or whitespace-only.")

    classifier = get_sentiment_classifier()
    results = classifier(stripped_text)
    
    if not results or not isinstance(results, list):
        raise RuntimeError("Sentiment pipeline returned invalid output structure.")

    first_result = results[0]
    raw_label = first_result.get("label", "").lower()
    score = float(first_result.get("score", 0.0))

    return {
        "label": raw_label,
        "score": score
    }


if __name__ == "__main__":
    test_sentences = [
        "I absolutely love this course!",
        "I am very disappointed with the service.",
        "I have a question about the course."
    ]

    print("--- Testing Sentiment Analyzer Module ---\n")
    for sentence in test_sentences:
        res = analyze_sentiment(sentence)
        print(f"Input: \"{sentence}\"")
        print(f"Detected Label: {res['label']} | Confidence Score: {res['score']:.4f}\n")
