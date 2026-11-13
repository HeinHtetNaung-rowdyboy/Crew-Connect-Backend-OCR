from functools import lru_cache
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')

@lru_cache
def settings():
    return {
        'backend': os.getenv('OCR_BACKEND', 'local').lower(),
        'use_ollama': os.getenv('USE_OLLAMA', 'false').lower() == 'true',
        'ollama_url': os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434'),
        'ollama_model': os.getenv('OLLAMA_MODEL', 'qwen2.5vl:3b'),
        'ollama_timeout': int(os.getenv('OLLAMA_TIMEOUT', '240')),
        'vision_zoom': float(os.getenv('VISION_ZOOM', '1.45')),
        'vision_max_pages': int(os.getenv('VISION_MAX_PAGES', '3')),
        'use_llm_classifier': os.getenv('USE_LLM_CLASSIFIER', 'true').lower() == 'true',
        'use_ai_reviewer': os.getenv('USE_AI_REVIEWER', 'true').lower() == 'true',
        'tesseract_lang': os.getenv('TESSERACT_LANG', 'eng'),
        'ocr_engine': os.getenv('OCR_ENGINE', 'tesseract').lower(),
        'ocr_fallback': os.getenv('OCR_FALLBACK', 'tesseract').lower(),
        # Cheap-to-expensive OCR cascade. Each engine runs only if the previous
        # result failed the validity gate. Empty value = legacy single engine
        # (OCR_ENGINE with OCR_FALLBACK).
        'ocr_cascade': [e.strip().lower() for e in os.getenv('OCR_CASCADE', 'tesseract,paddleocr').split(',') if e.strip()],
        'autocrop': os.getenv('OCR_AUTOCROP', 'true').lower() == 'true',
        'crop_max_area': float(os.getenv('OCR_CROP_MAX_AREA', '0.6')),
        'crop_max_outside': float(os.getenv('OCR_CROP_MAX_OUTSIDE', '0.05')),
        'ocr_crop_px': int(os.getenv('OCR_CROP_PX', '1100')),
        'vision_crop_px': int(os.getenv('VISION_CROP_PX', '1000')),
        'tesseract_zoom': float(os.getenv('TESSERACT_ZOOM', '2.0')),
        'paddle_zoom': float(os.getenv('PADDLE_ZOOM', '1.5')),
        'paddle_det_side': int(os.getenv('PADDLE_DET_SIDE', '960')),
        'paddle_cpu_threads': int(os.getenv('PADDLE_CPU_THREADS', '0')),
        # Empty = PaddleOCR default (heavier, more accurate). Mobile = faster.
        'paddle_det_model': os.getenv('PADDLE_DET_MODEL', 'PP-OCRv5_mobile_det'),
        'paddle_rec_model': os.getenv('PADDLE_REC_MODEL', 'PP-OCRv5_mobile_rec'),
        'paddleocr_lang': os.getenv('PADDLEOCR_LANG', 'en'),
        'ocr_min_text': int(os.getenv('OCR_MIN_TEXT', '100')),
        'max_upload_mb': int(os.getenv('MAX_UPLOAD_MB', '20')),
        'max_pages': int(os.getenv('MAX_PAGES', '30')),
        'confidence_threshold': float(os.getenv('CLASSIFICATION_THRESHOLD', '0.55')),
        'certificate_config': Path(os.getenv('CERTIFICATE_CONFIG', str(ROOT / 'data/certificate_setup.json'))),
        'landingai_key': os.getenv('VISION_AGENT_API_KEY', ''),
        'use_ai_extraction': os.getenv('LANDINGAI_EXTRACT', 'false').lower() == 'true',
    }