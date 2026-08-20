"""
Stage 2: Document pre-processing pipeline.
Handles deskewing, denoising, binarisation, and DPI normalisation
for both scanned and digital invoice images.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
from typing import Optional
import io


@dataclass
class PreprocessResult:
    image: np.ndarray           # processed image (BGR, uint8)
    pil_image: Image.Image      # PIL version for PaddleOCR
    original_size: tuple        # (width, height) before processing
    processed_size: tuple       # (width, height) after processing
    deskew_angle: float         # degrees of rotation applied
    was_binarized: bool


class InvoicePreprocessor:
    """
    Full pre-processing pipeline for invoice images.
    Input: path to image or numpy array
    Output: PreprocessResult with cleaned, deskewed, normalised image
    """

    TARGET_DPI = 300
    BINARIZE_BLOCK_SIZE = 31
    BINARIZE_C = 10

    def process(self, image_input) -> PreprocessResult:
        """
        Main entry point. Accepts:
        - str / Path: file path
        - np.ndarray: BGR image (from OpenCV)
        - bytes: raw image bytes
        """
        img = self._load(image_input)
        original_size = (img.shape[1], img.shape[0])
        logger.debug(f"Loaded image: {original_size[0]}x{original_size[1]}")

        img, orient_angle = self._auto_orient(img)
        img = self._normalise_dpi(img)
        img = self._denoise(img)
        img, angle = self._deskew(img)
        img, binarized = self._adaptive_binarize(img)
        img = self._remove_borders(img)
        img = self._sharpen(img)

        processed_size = (img.shape[1], img.shape[0])
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        logger.info(
            f"Pre-processing done: {original_size} → {processed_size}, "
            f"orient={orient_angle}°, deskew={angle:.2f}°, binarized={binarized}"
        )


        return PreprocessResult(
            image=img,
            pil_image=pil_img,
            original_size=original_size,
            processed_size=processed_size,
            deskew_angle=angle,
            was_binarized=binarized,
        )

    def process_minimal(self, image_input) -> PreprocessResult:
        """
        Lightweight pre-processing for images that come from digital PDFs.

        Digital PDF pages are already clean rasters — running denoise,
        binarisation, and sharpen on them *degrades* quality and wastes ~10s
        of CPU per page. This method only:
          1. DPI-normalises (upscale if too small for YOLO)
          2. Deskews (still useful if the scan was slightly rotated)

        Use this when the source PDF page was classified as digital by
        PDFConverter._is_digital_page().
        """
        img = self._load(image_input)
        original_size = (img.shape[1], img.shape[0])
        logger.debug(f"Minimal pre-process — Loaded image: {original_size[0]}x{original_size[1]}")

        img = self._normalise_dpi(img)
        img, angle = self._deskew(img)
        # ← no _denoise, no _adaptive_binarize, no _remove_borders, no _sharpen

        processed_size = (img.shape[1], img.shape[0])
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        logger.info(
            f"Minimal pre-processing done: {original_size} → {processed_size}, "
            f"deskew={angle:.2f}°, binarized=False (skipped — digital source)"
        )

        return PreprocessResult(
            image=img,
            pil_image=pil_img,
            original_size=original_size,
            processed_size=processed_size,
            deskew_angle=angle,
            was_binarized=False,
        )


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, source) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source.copy()
        if isinstance(source, bytes):
            arr = np.frombuffer(source, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image bytes")
            return img
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"OpenCV could not read: {path}")
        return img

    def _normalise_dpi(self, img: np.ndarray) -> np.ndarray:
        """
        Upscale images that are too small for reliable OCR.
        We target a minimum of 1200px on the long edge, which corresponds
        to roughly 300 DPI for a typical A4 invoice.
        """
        h, w = img.shape[:2]
        long_edge = max(h, w)
        if long_edge < 1200:
            scale = 1200 / long_edge
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.debug(f"Upscaled {w}x{h} → {new_w}x{new_h}")
        return img

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Fast Non-Local Means denoising — handles scanner noise well."""
        return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10,
                                                templateWindowSize=7,
                                                searchWindowSize=21)

    def _deskew(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Detects and corrects skew using Hough line transform on edges.
        Returns corrected image and angle applied.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            return img, 0.0

        angles = []
        for line in lines[:50]:  # use top 50 strongest lines
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if abs(angle) < 45:  # ignore near-vertical lines
                angles.append(angle)

        if not angles:
            return img, 0.0

        median_angle = float(np.median(angles))

        if abs(median_angle) < 0.3:  # skip trivial correction
            return img, 0.0

        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        logger.debug(f"Deskewed by {median_angle:.2f}°")
        return rotated, median_angle

    def _adaptive_binarize(self, img: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Converts to grayscale and applies adaptive thresholding.
        Only binarizes if the image appears to be a scan (low variance).
        Digital PDFs are kept as-is for better colour fidelity.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = gray.var()

        if variance > 2000:
            # High variance = likely a colour-rich digital invoice
            # Enhance contrast instead of full binarization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            return result, False

        # Low variance = scanned document — binarize aggressively
        binary = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=self.BINARIZE_BLOCK_SIZE,
            C=self.BINARIZE_C,
        )
        # Morphological cleanup to remove small specks
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return result, True

    def _remove_borders(self, img: np.ndarray) -> np.ndarray:
        """
        Crops out solid-colour borders (common in scanned images
        where the scanner bed edge appears as a thick black frame).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        # Add 10px padding
        x = max(0, x - 10)
        y = max(0, y - 10)
        w = min(img.shape[1] - x, w + 20)
        h = min(img.shape[0] - y, h + 20)
        return img[y:y + h, x:x + w]

    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        """Mild unsharp mask to improve OCR on slightly blurry text."""
        gaussian = cv2.GaussianBlur(img, (0, 0), 3)
        return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

    def _auto_orient(self, img: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Detects 90, 180, or 270 degree rotation on scanned documents using
        fast OCR text line orientation scoring on a small thumbnail.
        """
        h, w = img.shape[:2]
        # Use 800px thumbnail for reliable character recognition
        scale = 800 / max(h, w)
        thumb = cv2.resize(img, (int(w * scale), int(h * scale)))

        try:
            import easyocr
            if not hasattr(self, "_orient_reader"):
                self._orient_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

            best_angle = 0
            best_score = -1.0

            for angle in [0, 90, 180, 270]:
                if angle == 0:
                    rot = thumb
                elif angle == 90:
                    rot = cv2.rotate(thumb, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    rot = cv2.rotate(thumb, cv2.ROTATE_180)
                elif angle == 270:
                    rot = cv2.rotate(thumb, cv2.ROTATE_90_COUNTERCLOCKWISE)

                res = self._orient_reader.readtext(rot)
                score = sum(r[2] for r in res if len(r[1].strip()) >= 3 and r[2] > 0.35)

                if score > best_score:
                    best_score = score
                    best_angle = angle

            if best_angle != 0 and best_score > 3.0:
                logger.info(f"Auto-orienting scanned image by {best_angle}° (score={best_score:.2f})")
                if best_angle == 90:
                    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 90
                elif best_angle == 180:
                    return cv2.rotate(img, cv2.ROTATE_180), 180
                elif best_angle == 270:
                    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 270
        except Exception as e:
            logger.debug(f"Orientation auto-detection skipped: {e}")

        return img, 0


