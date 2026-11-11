"""Optional paid cloud OCR via official landingai-ade SDK v2."""
from pathlib import Path
from app.config import settings


def parse_pdf(path: Path) -> dict:
    cfg = settings()
    if not cfg['landingai_key']:
        raise ValueError('VISION_AGENT_API_KEY is required for LandingAI mode')
    try:
        from landingai_ade import LandingAIADE
    except ImportError as exc:
        raise RuntimeError('Install the optional dependency: pip install landingai-ade') from exc
    with LandingAIADE(apikey=cfg['landingai_key']) as client:
        parsed = client.v2.parse(document=path)
    # v2 markdown is one document; preserve raw text without inventing page mapping
    return {'text': parsed.markdown, 'pages': [{'page': 1, 'text': parsed.markdown}], 'engine': 'landingai'}


def extract_fields(markdown: str) -> dict:
    """Optional schema-guided cloud extraction. Charged separately from parsing."""
    cfg = settings()
    if not cfg['landingai_key']:
        raise ValueError('VISION_AGENT_API_KEY is required')
    from landingai_ade import LandingAIADE
    from app.extraction.schemas import CloudFields
    with LandingAIADE(apikey=cfg['landingai_key']) as client:
        result = client.v2.extract(schema=CloudFields, markdown=markdown)
    return result.extraction or {}
