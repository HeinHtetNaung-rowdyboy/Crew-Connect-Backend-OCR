import json
from app.config import settings
from app.classification.rules import evidence_for


def load_certificate_config():
    with settings()['certificate_config'].open(encoding='utf-8') as f:
        return json.load(f)['certificates']


def classify(text: str) -> dict:
    definitions = load_certificate_config()
    scored = []
    for item in definitions:
        hits, weight = evidence_for(text, item)
        if hits:
            scored.append((weight + 0.08*len(hits), item, hits))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return {'certificate_name': 'Unknown', 'certificate_type': None,
                'category': None, 'confidence': 0.0, 'matched_aliases': [], 'requires_review': True}
    top, item, hits = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0
    # Distinctive multi-part document structures give strongest evidence.
    confidence = round(min(.99, .5 + .16*len(hits) + min(top-runner, 1)*.13), 2)
    ambiguous = bool(runner and (top-runner)/max(top, .001) < .20)
    return {'certificate_name': item['name'], 'certificate_type': item['certificate_type'],
            'category': item['category'], 'confidence': confidence, 'matched_aliases': hits,
            'requires_review': confidence < settings()['confidence_threshold'] or ambiguous}
