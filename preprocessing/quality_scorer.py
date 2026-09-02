"""
preprocessing/quality_scorer.py

Phase 2: Document Quality Scoring Engine.
Computes multi-dimensional image & document quality metrics:
1. Sharpness / Blur estimation via Laplacian variance (Var(ΔI))
2. Brightness & Contrast distribution (histogram spread and dynamic range)
3. Background skew & noise estimation
4. Text density estimation
5. Composite Quality Score [0.0 - 1.0]

Directs the pipeline on whether lightweight or enhanced adaptive preprocessing is required.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path
from loguru import logger
from PIL import Image


@dataclass
class QualityAssessment:
    composite_score: float          # [0.0, 1.0] (1.0 = pristine digital, < 0.65 = degraded)
    blur_score: float               # Raw Laplacian variance (> 100 = sharp)
    is_blurry: bool
    contrast_score: float           # RMS contrast [0.0, 1.0]
    is_low_contrast: bool
    mean_brightness: float          # [0, 255] (ideal 180-240 for white background)
    is_poor_lighting: bool
    estimated_skew_angle: float     # Estimated skew in degrees
    text_density: float             # Foreground pixel ratio [0.0, 1.0]
    is_acceptable: bool             # Whether document is readable
    recommended_actions: list[str] = field(default_factory=list)
    has_ruled_lines: bool = False
    ink_strength: float = 1.0       # [0.0, 1.0] foreground ink contrast
    background_uniformity: float = 1.0 # [0.0, 1.0] paper texture cleanliness
    estimated_ink_type: str = "black_ink" # "blue_ink" | "black_ink" | "pencil" | "carbon_copy" | "mixed"


class DocumentQualityScorer:
    """
    Evaluates raw images / PDF rasterizations before OCR to prevent
    wasting expensive models on corrupted or severely distorted inputs.
    """

    BLUR_THRESHOLD = 80.0           # Laplacian variance below 80 indicates severe blur
    CONTRAST_THRESHOLD = 0.20       # Normalized RMS contrast below 0.20 indicates washed out scan
    BRIGHTNESS_MIN = 70.0           # Below 70 = severe underexposure / dark photo
    BRIGHTNESS_MAX = 250.0          # Above 250 = severe overexposure / blown highlights

    def assess(self, image_input: Union[np.ndarray, Image.Image, str, Path, bytes]) -> QualityAssessment:
        img = self._to_cv2(image_input)
        h, w = img.shape[:2]

        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 1. Blur score (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_var = float(laplacian.var())
        is_blurry = blur_var < self.BLUR_THRESHOLD

        # 2. Brightness & RMS Contrast
        mean_brightness = float(np.mean(gray))
        rms_contrast = float(np.std(gray) / 255.0)
        is_low_contrast = rms_contrast < self.CONTRAST_THRESHOLD
        is_poor_lighting = mean_brightness < self.BRIGHTNESS_MIN or mean_brightness > self.BRIGHTNESS_MAX

        # 3. Estimated Skew Angle
        skew_angle = self._estimate_skew(gray)

        # 4. Text density (Otsu foreground ratio) & Ruled lines
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fg_pixels = cv2.countNonZero(thresh)
        text_density = float(fg_pixels / (w * h)) if (w * h) > 0 else 0.0

        kw = max(20, int(w * 0.08))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
        h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ruled_lines = [c for c in h_contours if cv2.boundingRect(c)[2] > (w * 0.20)]
        has_ruled_lines = len(ruled_lines) >= 3

        # 5. Ink Strength & Background Uniformity
        fg_mask = thresh > 0
        bg_mask = thresh == 0
        mean_fg = float(np.mean(gray[fg_mask])) if np.any(fg_mask) else 0.0
        mean_bg = float(np.mean(gray[bg_mask])) if np.any(bg_mask) else 255.0
        std_bg = float(np.std(gray[bg_mask])) if np.any(bg_mask) else 0.0

        ink_strength = round(max(0.0, min(1.0, (mean_bg - mean_fg) / max(1.0, mean_bg))), 3)
        background_uniformity = round(max(0.0, min(1.0, 1.0 - (std_bg / 100.0))), 3)

        # 6. Estimate Ink Type
        estimated_ink_type = "black_ink"
        if len(img.shape) == 3 and np.any(fg_mask):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            fg_hsv = hsv[fg_mask]
            if len(fg_hsv) > 0:
                mean_hue = np.median(fg_hsv[:, 0])
                mean_sat = np.median(fg_hsv[:, 1])
                mean_val = np.median(fg_hsv[:, 2])

                # Blue ink in OpenCV HSV (H: 90-130) with decent saturation
                if 90 <= mean_hue <= 135 and mean_sat > 40:
                    estimated_ink_type = "blue_ink"
                elif mean_sat < 30 and 80 < mean_val < 180:
                    estimated_ink_type = "pencil"
                elif mean_sat < 30 and mean_val <= 80:
                    estimated_ink_type = "black_ink"
                elif mean_sat < 40 and std_bg > 35:
                    estimated_ink_type = "carbon_copy"
                else:
                    estimated_ink_type = "mixed"

        # 7. Recommended Actions & Composite Quality Scoring
        actions = []
        blur_factor = min(1.0, blur_var / 300.0)
        contrast_factor = min(1.0, rms_contrast / 0.40)
        brightness_factor = 1.0 - (abs(mean_brightness - 200.0) / 200.0)
        brightness_factor = max(0.0, min(1.0, brightness_factor))

        if is_blurry:
            actions.append("apply_unsharp_mask")
        if is_low_contrast:
            actions.append("apply_clahe_contrast_enhancement")
        if is_poor_lighting:
            actions.append("apply_illumination_normalization")
        if abs(skew_angle) > 0.8:
            actions.append("apply_deskew")
        if has_ruled_lines:
            actions.append("remove_ruled_lines")
        if estimated_ink_type in ("blue_ink", "pencil", "carbon_copy"):
            actions.append(f"enhance_{estimated_ink_type}")

        composite = round(
            0.40 * blur_factor +
            0.35 * contrast_factor +
            0.25 * brightness_factor,
            3
        )

        is_acceptable = composite >= 0.35 and not (is_blurry and is_low_contrast)

        return QualityAssessment(
            composite_score=composite,
            blur_score=round(blur_var, 2),
            is_blurry=is_blurry,
            contrast_score=round(rms_contrast, 3),
            is_low_contrast=is_low_contrast,
            mean_brightness=round(mean_brightness, 1),
            is_poor_lighting=is_poor_lighting,
            estimated_skew_angle=round(skew_angle, 2),
            text_density=round(text_density, 4),
            is_acceptable=is_acceptable,
            recommended_actions=actions,
            has_ruled_lines=has_ruled_lines,
            ink_strength=ink_strength,
            background_uniformity=background_uniformity,
            estimated_ink_type=estimated_ink_type,
        )

    def _estimate_skew(self, gray: np.ndarray) -> float:
        """Fast edge-based Hough line skew estimator."""
        try:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
            if lines is None:
                return 0.0
            angles = []
            for line in lines[:30]:
                theta = line[0][1]
                deg = np.degrees(theta) - 90
                if abs(deg) < 45:
                    angles.append(deg)
            return float(np.median(angles)) if angles else 0.0
        except Exception:
            return 0.0

    def _to_cv2(self, inp) -> np.ndarray:
        if isinstance(inp, np.ndarray):
            return inp
        if isinstance(inp, Image.Image):
            return cv2.cvtColor(np.array(inp), cv2.COLOR_RGB2BGR)
        if isinstance(inp, (str, Path)):
            img = cv2.imread(str(inp))
            if img is None:
                raise ValueError(f"Could not read image from {inp}")
            return img
        if isinstance(inp, bytes):
            arr = np.frombuffer(inp, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image bytes")
            return img
        raise TypeError(f"Unsupported image input type: {type(inp)}")
