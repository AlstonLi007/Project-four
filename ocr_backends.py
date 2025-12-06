from __future__ import annotations

"""Pluggable OCR backends used by :mod:`vision_controller`.

Two engines are provided out of the box:

* :class:`TesseractOcrEngine` wraps ``pytesseract`` for lightweight OCR.
* :class:`PaddleOcrEngine` wraps ``paddleocr`` for multi-language, higher-quality OCR.

Both expose a common ``recognize`` method so callers can swap engines without
changing application code.
"""

from typing import Optional, Protocol

import numpy as np


class OcrEngine(Protocol):
    """Protocol describing the minimal OCR API used by the vision stack."""

    def recognize(self, image: np.ndarray, lang: Optional[str] = None) -> str:
        """Return recognized text from the given image."""
        ...


class TesseractOcrEngine:
    """Lightweight OCR engine backed by ``pytesseract``."""

    def __init__(self, lang: str = "eng") -> None:
        import pytesseract

        self._tesseract = pytesseract
        self.default_lang = lang

    def recognize(self, image: np.ndarray, lang: Optional[str] = None) -> str:
        language = lang or self.default_lang
        return self._tesseract.image_to_string(image, lang=language)


class PaddleOcrEngine:
    """OCR engine backed by :mod:`paddleocr` for richer multilingual support."""

    def __init__(self, lang: str = "ch", **kwargs) -> None:
        # ``paddleocr`` is optional; import inside to keep it optional at runtime.
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, **kwargs)

    def recognize(self, image: np.ndarray, lang: Optional[str] = None) -> str:
        # The PaddleOCR instance is typically language-specific; ``lang`` is ignored
        # here to keep the interface consistent.
        result = self._ocr.ocr(image, cls=True)
        lines: list[str] = []
        for page in result:
            for _box, (text, _score) in page:
                cleaned = text.strip()
                if cleaned:
                    lines.append(cleaned)
        return "\n".join(lines)
