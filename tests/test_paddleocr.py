from unittest.mock import patch
from app.ocr.pdf_parser import ocr_image


def test_ocr_prefers_paddle():
    cfg={'ocr_engine':'paddleocr','ocr_fallback':'tesseract','paddleocr_lang':'en','tesseract_lang':'eng'}
    with patch('app.ocr.pdf_parser._paddle_ocr', return_value='SID') as paddle:
        assert ocr_image(object(), cfg)==('SID','paddleocr')
        paddle.assert_called_once()


def test_fallback_to_tesseract():
    cfg={'ocr_engine':'paddleocr','ocr_fallback':'tesseract','paddleocr_lang':'en','tesseract_lang':'eng'}
    with patch('app.ocr.pdf_parser._paddle_ocr', side_effect=RuntimeError('missing model')):
        with patch('app.ocr.pdf_parser._tesseract_ocr', return_value='Seaman Book'):
            assert ocr_image(object(), cfg)==('Seaman Book','tesseract')
