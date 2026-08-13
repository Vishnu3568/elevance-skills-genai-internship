"""Response Policy module for Customer Service Chatbot.

Defines the confidence threshold and pure response policy function.
"""

HIGH_CONFIDENCE_THRESHOLD = 0.80


def apply_response_policy(answer: str, sentiment_label: str, score: float) -> str:
    """Apply confidence-aware sentiment response policy to a RAG answer.

    Policy Hierarchy:
    1. OOD protection: If answer is 'I don't know.', return unmodified.
    2. High-confidence negative (>= 0.80): Add empathy + customer service escalation notice.
    3. Standard negative (< 0.80): Add standard empathy notice.
    4. Positive: Add appreciative feedback notice.
    5. Neutral: Return unmodified.
    """
    norm_ans = answer.strip().lower().rstrip(".")
    is_ood = norm_ans in ("i don't know", "i do not know") or norm_ans.startswith("i don't know")

    if not is_ood:
        if sentiment_label == "NEGATIVE":
            if score >= HIGH_CONFIDENCE_THRESHOLD:
                return f"I am sorry to hear about your experience and frustration. Your concern may benefit from additional support from our customer service team. {answer}"
            else:
                return f"I am sorry to hear about your experience and frustration. {answer}"
        elif sentiment_label == "POSITIVE":
            return f"Thank you for your positive feedback! {answer}"

    return answer
