from app.classification.similarity import normalize


def evidence_for(doc: str, item: dict):
    text = normalize(doc)
    matches = [alias for alias in item.get('aliases', []) if normalize(alias) and normalize(alias) in text]
    positive = [token for token in item.get('required_any', []) if normalize(token) in text]
    negatives = [token for token in item.get('exclude_if', []) if normalize(token) in text]
    if negatives or (item.get('required_any') and not positive):
        return [], 0.0
    weight = sum(min(len(normalize(m)), 85) / 85 for m in matches)
    if positive:
        weight += 0.4
    return matches, weight
