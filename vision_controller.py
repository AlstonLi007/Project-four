from __future__ import annotations

"""Vision + mouse/keyboard controller for template matching and OCR automation.

This module wraps screen capture, template matching, OCR, and basic mouse/keyboard
controls to support vision-driven RPA flows. It is intentionally lightweight and
configurable so tasks can compose it with different templates and regions.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pyautogui
from mss import mss

from ocr_backends import OcrEngine, TesseractOcrEngine


@dataclass
class Region:
    """Rectangular screen region in absolute pixel coordinates."""

    left: int
    top: int
    width: int
    height: int

    def to_mss(self) -> dict:
        """Return an ``mss``-compatible region mapping."""

        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


class VisionController:
    """Capture the screen, locate templates, OCR regions, and drive input events."""

    def __init__(
        self,
        template_dir: str = "templates",
        monitor_index: int = 1,
        ocr_lang: str = "eng",
        debug: bool = False,
        ocr_engine: Optional[OcrEngine] = None,
    ) -> None:
        """Initialize the controller.

        Args:
            template_dir: Directory containing template PNGs.
            monitor_index: Index into ``mss.monitors`` (1 is primary screen).
            ocr_lang: Language used for Tesseract OCR.
            debug: When True, prints matching/OCR diagnostics.
        """

        self.template_dir = template_dir
        self.monitor_index = monitor_index
        self.ocr_lang = ocr_lang
        self.debug = debug
        self.ocr_engine: OcrEngine = ocr_engine or TesseractOcrEngine(lang=ocr_lang)

        # Optional: configure tesseract executable path if needed on Windows.
        # If you want to use Tesseract with a non-default install location, set
        # ``pytesseract.pytesseract.tesseract_cmd`` on the chosen engine.

        # Slow down actions slightly to appear less mechanical and avoid misses.
        pyautogui.PAUSE = 0.1

    # -------- Screen capture --------

    def grab_screen(self) -> np.ndarray:
        """Capture the full monitor as a BGR numpy array."""

        with mss() as sct:
            monitor = sct.monitors[self.monitor_index]
            img = np.array(sct.grab(monitor))
        return img[:, :, :3]  # Drop alpha.

    def grab_region(self, region: Region) -> np.ndarray:
        """Capture a rectangular region as BGR numpy array."""

        with mss() as sct:
            img = np.array(sct.grab(region.to_mss()))
        return img[:, :, :3]

    # -------- Template matching --------

    def _load_template(self, name_or_path: str) -> np.ndarray:
        """Load a template image from ``template_dir`` or absolute path."""

        if os.path.isfile(name_or_path):
            path = name_or_path
        else:
            path = os.path.join(self.template_dir, name_or_path)
        tpl = cv2.imread(path, cv2.IMREAD_COLOR)
        if tpl is None:
            raise FileNotFoundError(f"Template not found: {path}")
        return tpl

    def locate_template(
        self, template_name: str, threshold: float = 0.9
    ) -> Optional[Tuple[int, int, float]]:
        """Find a template on the current screen.

        Returns the center coordinates (cx, cy) and match score or ``None`` if
        below ``threshold``.
        """

        screen = self.grab_screen()
        tpl = self._load_template(template_name)

        res = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if self.debug:
            print(f"[vision] match {template_name}: max_val={max_val:.4f}")

        if max_val < threshold:
            return None

        h, w = tpl.shape[:2]
        x, y = max_loc
        cx, cy = x + w // 2, y + h // 2
        return cx, cy, max_val

    def click_template(
        self,
        template_name: str,
        threshold: float = 0.9,
        move_duration: float = 0.2,
        click_delay: float = 0.1,
    ) -> bool:
        """Locate a template and click its center if found."""

        located = self.locate_template(template_name, threshold=threshold)
        if not located:
            if self.debug:
                print(f"[vision] {template_name} not found (threshold={threshold})")
            return False

        cx, cy, score = located
        if self.debug:
            print(f"[vision] clicking {template_name} at ({cx}, {cy}), score={score:.3f}")

        pyautogui.moveTo(cx, cy, duration=move_duration)
        time.sleep(click_delay)
        pyautogui.click()
        return True

    # -------- OCR --------

    def read_region(
        self,
        region: Region,
        lang: Optional[str] = None,
        preprocess: bool = True,
    ) -> str:
        """OCR a region using the configured :class:`OcrEngine`."""

        img = self.grab_region(region)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if preprocess:
            gray = cv2.resize(gray, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_LINEAR)
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

        lang = lang or self.ocr_lang
        text = self.ocr_engine.recognize(gray, lang=lang)
        if self.debug:
            print(f"[vision] OCR text preview: {repr(text[:200])}")
        return text

    # -------- Mouse / keyboard helpers --------

    def click(self, x: int, y: int, duration: float = 0.2) -> None:
        """Move to a coordinate and click."""

        pyautogui.moveTo(x, y, duration=duration)
        pyautogui.click()

    def scroll(self, clicks: int) -> None:
        """Scroll the mouse wheel (positive is up, negative is down)."""

        pyautogui.scroll(clicks)

    def type_text(self, text: str, interval: float = 0.02) -> None:
        """Type text with a configurable keystroke interval."""

        pyautogui.typewrite(text, interval=interval)

    def press_key(self, key: str) -> None:
        """Press a single key (e.g., ``'esc'`` or ``'enter'``)."""

        pyautogui.press(key)
