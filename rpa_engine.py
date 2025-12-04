from __future__ import annotations

"""Vision-driven RPA skeleton for screen-based automation.

This module provides a minimal building block for a "look-at-screen, move mouse" workflow
that can be configured with template images and state-machine steps. It is intentionally
framework-light so it can slot into the existing WeChat assistant codebase and share the
SQLite logging pipeline if desired.

Key pieces:
    - :class:`ScreenCapture` grabs screenshots using ``mss``.
    - :class:`TemplateMatcher` performs OpenCV template matching to locate UI affordances.
    - :class:`VisionController` wraps template lookup, OCR reading, and mouse actions
      via ``pyautogui``.
    - :class:`VisualCrawler` shows how to drive a simple finite-state machine using the
      vision helpers.

Dependencies (install as needed):
    pip install mss opencv-python pyautogui pytesseract numpy

Notes:
    * This file does not execute anything on import; consumers should construct the
      classes below and call ``VisualCrawler.run`` inside their own process.
    * OCR requires a local Tesseract installation available on PATH.
    * All coordinates are absolute screen coordinates; supply ``region`` hints in
      template specs to limit search to sub-areas for speed/stability.
"""

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import pyautogui
import time
import pytesseract
from mss import mss


@dataclass
class TemplateSpec:
    """Configuration for locating a UI element by template matching.

    Attributes:
        name: Logical label for the template (e.g., "next_page").
        path: Filesystem path to the template image.
        threshold: Minimum normalized match score to accept (0.0-1.0).
        region: Optional search region as (x, y, width, height). If provided, only that
            sub-rectangle of the screen is searched to reduce false positives.
    """

    name: str
    path: Path
    threshold: float = 0.9
    region: Optional[Tuple[int, int, int, int]] = None


@dataclass
class TemplateMatch:
    """Result of a template search."""

    name: str
    center: Tuple[int, int]
    score: float
    region: Optional[Tuple[int, int, int, int]]


class ScreenCapture:
    """Grabs screenshots using mss and returns BGR numpy arrays."""

    def __init__(self) -> None:
        self._sct = mss()

    def grab(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Capture the full screen or a specific region.

        Args:
            region: Optional (x, y, width, height) tuple.
        Returns:
            A numpy array in BGR format compatible with OpenCV.
        """

        if region:
            x, y, w, h = region
            monitor = {"left": x, "top": y, "width": w, "height": h}
        else:
            monitor = self._sct.monitors[1]
        shot = np.array(self._sct.grab(monitor))
        # mss returns BGRA; drop alpha to BGR.
        return shot[:, :, :3]


class TemplateMatcher:
    """Performs template matching on screenshots."""

    def __init__(self, templates: Iterable[TemplateSpec]):
        self.templates: Dict[str, TemplateSpec] = {tpl.name: tpl for tpl in templates}

    def locate(self, screen_bgr: np.ndarray, name: str) -> Optional[TemplateMatch]:
        """Find a template on the provided screenshot.

        Args:
            screen_bgr: Screenshot array in BGR format.
            name: Template name to search for.
        Returns:
            TemplateMatch if score exceeds threshold, else None.
        """

        spec = self.templates.get(name)
        if spec is None:
            raise KeyError(f"unknown template: {name}")

        tpl = cv2.imread(str(spec.path), cv2.IMREAD_COLOR)
        if tpl is None:
            raise FileNotFoundError(f"template image missing: {spec.path}")

        search_img = screen_bgr
        origin = (0, 0)
        if spec.region:
            x, y, w, h = spec.region
            search_img = screen_bgr[y : y + h, x : x + w]
            origin = (x, y)

        res = cv2.matchTemplate(search_img, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < spec.threshold:
            return None

        h, w = tpl.shape[:2]
        top_left_x = origin[0] + max_loc[0]
        top_left_y = origin[1] + max_loc[1]
        center = (top_left_x + w // 2, top_left_y + h // 2)
        return TemplateMatch(name=spec.name, center=center, score=float(max_val), region=spec.region)


class VisionController:
    """Wraps screen capture, template matching, OCR, and mouse actions."""

    def __init__(self, matcher: TemplateMatcher, capture: Optional[ScreenCapture] = None) -> None:
        self.matcher = matcher
        self.capture = capture or ScreenCapture()

    def _grab(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        return self.capture.grab(region=region)

    def locate(self, name: str) -> Optional[TemplateMatch]:
        screen = self._grab(self.matcher.templates[name].region)
        return self.matcher.locate(screen, name)

    def click(self, name: str, move_duration: float = 0.2) -> Optional[TemplateMatch]:
        """Find a template and click its center."""

        screen = self._grab(self.matcher.templates[name].region)
        match = self.matcher.locate(screen, name)
        if match is None:
            return None
        cx, cy = match.center
        pyautogui.moveTo(cx, cy, duration=move_duration)
        pyautogui.click()
        return match

    def ocr_region(self, region: Tuple[int, int, int, int]) -> str:
        """Run OCR on a screen subregion (x, y, w, h)."""

        img = self._grab(region)
        text = pytesseract.image_to_string(img)
        return text.strip()

    def read_text_near(self, name: str, padding: int = 10) -> Optional[str]:
        """OCR a small box around a located template."""

        match = self.locate(name)
        if match is None:
            return None
        tpl = cv2.imread(str(self.matcher.templates[name].path), cv2.IMREAD_COLOR)
        assert tpl is not None
        h, w = tpl.shape[:2]
        cx, cy = match.center
        x = max(cx - w // 2 - padding, 0)
        y = max(cy - h // 2 - padding, 0)
        region = (x, y, w + 2 * padding, h + 2 * padding)
        return self.ocr_region(region)


class CrawlerState(Enum):
    INIT = auto()
    LIST_PAGE = auto()
    ITEM_OPEN = auto()
    DATA_EXTRACT = auto()
    NEXT_PAGE = auto()
    DONE = auto()


class VisualCrawler:
    """Minimal state-machine driver that uses the vision helpers.

    Customize the callbacks to suit the target application; this class provides a
    deterministic heartbeat loop and a few convenience hooks for controlling the
    mouse/keyboard based on visual cues.
    """

    def __init__(self, controller: VisionController) -> None:
        self.controller = controller
        self.state = CrawlerState.INIT
        self.page_index = 0

    def on_init(self) -> None:
        if self.controller.locate("app_logo"):
            self.state = CrawlerState.LIST_PAGE

    def on_list_page(self) -> None:
        match = self.controller.click("first_item")
        if match:
            self.state = CrawlerState.ITEM_OPEN

    def on_item_open(self) -> None:
        if self.controller.locate("content_loaded"):
            self.state = CrawlerState.DATA_EXTRACT

    def on_data_extract(self) -> None:
        # Replace this with your own extraction routine (e.g., OCR a region and store).
        self.state = CrawlerState.NEXT_PAGE

    def on_next_page(self) -> None:
        match = self.controller.click("next_page")
        if match:
            self.page_index += 1
            self.state = CrawlerState.LIST_PAGE
        else:
            self.state = CrawlerState.DONE

    def step(self) -> None:
        if self.state == CrawlerState.INIT:
            self.on_init()
        elif self.state == CrawlerState.LIST_PAGE:
            self.on_list_page()
        elif self.state == CrawlerState.ITEM_OPEN:
            self.on_item_open()
        elif self.state == CrawlerState.DATA_EXTRACT:
            self.on_data_extract()
        elif self.state == CrawlerState.NEXT_PAGE:
            self.on_next_page()

    def run(self, max_steps: int = 500, sleep_seconds: float = 0.3) -> None:
        for _ in range(max_steps):
            if self.state == CrawlerState.DONE:
                break
            self.step()
            time.sleep(sleep_seconds)


def demo_templates(base_dir: Path) -> Iterable[TemplateSpec]:
    """Helper to build TemplateSpec instances from a folder.

    Expects files like ``app_logo.png``, ``first_item.png``, ``content_loaded.png``,
    and ``next_page.png`` inside the provided directory.
    """

    names = ["app_logo", "first_item", "content_loaded", "next_page"]
    for name in names:
        yield TemplateSpec(name=name, path=base_dir / f"{name}.png")


def main() -> None:
    """Example usage.

    Adjust the template directory and thresholds to match your target UI before running.
    """

    templates = list(demo_templates(Path("templates")))
    matcher = TemplateMatcher(templates)
    controller = VisionController(matcher)
    crawler = VisualCrawler(controller)
    crawler.run()


if __name__ == "__main__":
    main()
