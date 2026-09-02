"""
validation/char_confusion.py

Field-type-aware character confusion correction engine for handwriting and low-quality OCR.
Maintains known bidirectional character confusions:
    0 <-> O, o, Q
    1 <-> I, l, |, i
    2 <-> Z, z
    5 <-> S, s
    6 <-> G, b
    8 <-> B
    9 <-> g, q

Applies domain constraints (e.g. numeric fields, GSTIN positions, IFSC formats, Dates)
to convert optical character confusions into valid, structurally verified candidates.
"""

from __future__ import annotations
import re
from datetime import datetime
from typing import Optional

# Bidirectional confusion mappings
LETTER_TO_DIGIT = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "|": "1", "i": "1", "!": "1", "T": "1",
    "Z": "2", "z": "2",
    "E": "3",
    "A": "4",
    "S": "5", "s": "5", "$": "5",
    "G": "6", "b": "6",
    "T": "7", "/": "7",
    "B": "8", "&": "8",
    "g": "9", "q": "9", "P": "9",
}

DIGIT_TO_LETTER = {
    "0": ["O", "D", "Q"],
    "1": ["I", "L", "T"],
    "2": ["Z"],
    "3": ["E"],
    "4": ["A"],
    "5": ["S"],
    "6": ["G"],
    "7": ["T", "Z"],
    "8": ["B"],
    "9": ["g", "q"],
}

# Modulo 36 GSTIN Checksum calculation
GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_CHAR_MAP = {c: i for i, c in enumerate(GSTIN_CHARS)}


def verify_gstin_checksum(gstin: Optional[str]) -> bool:
    """Validate GSTIN format and modulo-36 check digit."""
    if not gstin or len(gstin.strip()) != 15:
        return False
    gstin = gstin.strip().upper()
    if not re.match(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$", gstin):
        return False
    try:
        factor = 1
        total = 0
        for i in range(14):
            code_pt = GSTIN_CHAR_MAP[gstin[i]]
            addend = factor * code_pt
            factor = 2 if factor == 1 else 1
            addend = (addend // 36) + (addend % 36)
            total += addend
        remainder = total % 36
        check_code = (36 - remainder) % 36
        return GSTIN_CHARS[check_code] == gstin[14]
    except Exception:
        return False


def calculate_gstin_checksum_digit(gstin_14: str) -> Optional[str]:
    """Given 14 characters of a GSTIN, computes the 15th checksum character."""
    if len(gstin_14) != 14:
        return None
    try:
        factor = 1
        total = 0
        for i in range(14):
            code_pt = GSTIN_CHAR_MAP[gstin_14[i]]
            addend = factor * code_pt
            factor = 2 if factor == 1 else 1
            addend = (addend // 36) + (addend % 36)
            total += addend
        remainder = total % 36
        check_code = (36 - remainder) % 36
        return GSTIN_CHARS[check_code]
    except Exception:
        return None


def correct_numeric_field(raw: str, allow_decimal: bool = True) -> list[tuple[str, float]]:
    """
    Given a raw OCR token extracted for a numeric field (e.g. quantity, amount, rate),
    applies letter->digit confusions and returns ranked (candidate, confidence) pairs.
    """
    if not raw or not str(raw).strip():
        return []

    clean = str(raw).strip().replace("₹", "").replace("Rs.", "").replace("INR", "").replace(",", "")
    clean = re.sub(r"\s+", "", clean)

    candidates: list[tuple[str, float]] = []

    # 1. Direct parse test
    try:
        val = float(clean)
        formatted = f"{val:.2f}" if allow_decimal and "." in clean else f"{int(val)}" if not allow_decimal else str(clean)
        candidates.append((formatted, 0.95))
    except ValueError:
        pass

    # 2. Convert confused characters to digits
    substituted = []
    num_substitutions = 0
    dot_seen = False
    valid = True

    for ch in clean:
        if ch.isdigit():
            substituted.append(ch)
        elif ch in (".", ",") and allow_decimal:
            if not dot_seen:
                substituted.append(".")
                dot_seen = True
            else:
                valid = False
                break
        elif ch in LETTER_TO_DIGIT:
            substituted.append(LETTER_TO_DIGIT[ch])
            num_substitutions += 1
        else:
            # Unrecognized non-numeric character
            valid = False
            break

    if valid and substituted:
        sub_str = "".join(substituted).strip(".")
        if sub_str:
            try:
                val = float(sub_str)
                # Score degrades with number of character replacements
                conf = max(0.50, 0.90 - (0.10 * num_substitutions))
                cand_str = f"{val:.2f}" if allow_decimal and "." in sub_str else str(int(val)) if not allow_decimal else sub_str
                if not any(c[0] == cand_str for c in candidates):
                    candidates.append((cand_str, round(conf, 2)))
            except ValueError:
                pass

    return sorted(candidates, key=lambda x: x[1], reverse=True)


def correct_gstin(raw: str) -> list[tuple[str, float]]:
    """
    Takes an OCR string intended to be a GSTIN and applies structure-aware
    character corrections based on standard 15-character Indian GSTIN positions:
      Pos 0-1: 2 digits (State code, e.g. 27, 07, 19)
      Pos 2-6: 5 uppercase letters (PAN Entity)
      Pos 7-10: 4 digits (PAN Number)
      Pos 11: 1 uppercase letter (PAN Check letter)
      Pos 12: 1 alphanumeric (Entity number, e.g. 1, 2, Z)
      Pos 13: 'Z' (Fixed default character)
      Pos 14: 1 alphanumeric (Checksum digit)
    """
    if not raw:
        return []

    clean = re.sub(r"[^A-Za-z0-9]", "", str(raw)).upper()
    if len(clean) != 15:
        # Try extracting 15 contiguous or semi-contiguous chars
        if len(clean) > 15:
            clean = clean[:15]
        else:
            return []

    # Check if direct match with checksum
    candidates: list[tuple[str, float]] = []
    if verify_gstin_checksum(clean):
        candidates.append((clean, 0.99))
        return candidates

    # Apply positional corrections
    corrected = list(clean)
    changes = 0

    # Pos 0, 1 -> Digits
    for i in (0, 1):
        if not corrected[i].isdigit() and corrected[i] in LETTER_TO_DIGIT:
            corrected[i] = LETTER_TO_DIGIT[corrected[i]]
            changes += 1

    # Pos 2..6 -> Letters
    for i in range(2, 7):
        if corrected[i].isdigit():
            possible_letters = DIGIT_TO_LETTER.get(corrected[i], ["O"])
            corrected[i] = possible_letters[0]
            changes += 1

    # Pos 7..10 -> Digits
    for i in range(7, 11):
        if not corrected[i].isdigit() and corrected[i] in LETTER_TO_DIGIT:
            corrected[i] = LETTER_TO_DIGIT[corrected[i]]
            changes += 1

    # Pos 11 -> Letter
    if corrected[11].isdigit():
        possible_letters = DIGIT_TO_LETTER.get(corrected[11], ["A"])
        corrected[11] = possible_letters[0]
        changes += 1

    # Pos 12 -> Entity number (typically '1', '2', etc.)
    if corrected[12] in ("I", "l", "|"):
        corrected[12] = "1"
        changes += 1
    elif corrected[12] == "O":
        corrected[12] = "0"
        changes += 1

    # Pos 13 -> Almost always 'Z'
    if corrected[13] != "Z":
        if corrected[13] in ("2", "7", "s", "S", "z"):
            corrected[13] = "Z"
            changes += 1

    corrected_str = "".join(corrected)

    # 1. Test if positional correction passes checksum
    if verify_gstin_checksum(corrected_str):
        conf = max(0.65, 0.95 - (0.08 * changes))
        candidates.append((corrected_str, round(conf, 2)))
    else:
        # 2. Try recalculating the 15th checksum character
        calc_check = calculate_gstin_checksum_digit(corrected_str[:14])
        if calc_check:
            rec_gstin = corrected_str[:14] + calc_check
            if verify_gstin_checksum(rec_gstin):
                conf = max(0.60, 0.90 - (0.08 * (changes + 1)))
                candidates.append((rec_gstin, round(conf, 2)))

    return sorted(candidates, key=lambda x: x[1], reverse=True)


def correct_ifsc(raw: str) -> list[tuple[str, float]]:
    """
    Corrects IFSC code formatting:
    Standard structure: 4 letters (Bank), '0' (5th char), 6 alphanumeric (Branch).
    E.g., "SBINOOOO123" -> "SBIN0000123"
    """
    if not raw:
        return []
    clean = re.sub(r"[^A-Za-z0-9]", "", str(raw)).upper()
    if len(clean) != 11:
        return []

    candidates: list[tuple[str, float]] = []

    # Map first 4 to letters, 5th to '0'
    corrected = list(clean)
    changes = 0

    for i in range(4):
        if corrected[i].isdigit():
            corrected[i] = DIGIT_TO_LETTER.get(corrected[i], ["O"])[0]
            changes += 1

    if corrected[4] != "0":
        corrected[4] = "0"
        changes += 1

    # Candidate A: With 'O' -> '0' in branch code (most Indian banks use 6 numeric digits e.g. SBIN0000123)
    corrected_b = list(corrected)
    changes_b = changes
    for i in range(5, 11):
        if corrected_b[i] == "O":
            corrected_b[i] = "0"
            changes_b += 1
    cand_b = "".join(corrected_b)
    if re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", cand_b):
        conf_b = max(0.65, 0.95 - (0.03 * changes_b))
        candidates.append((cand_b, round(conf_b, 2)))

    # Candidate B: Raw branch code preserved (lower priority if it contains 'O' that could be '0')
    cand_a = "".join(corrected)
    if cand_a != cand_b and re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", cand_a):
        conf = max(0.50, 0.75 - (0.05 * changes))
        candidates.append((cand_a, round(conf, 2)))

    return sorted(candidates, key=lambda x: x[1], reverse=True)


def correct_date(raw: str) -> list[tuple[str, float]]:
    """
    Corrects common optical errors in date strings:
    E.g. "I5/O7/2O26" -> "15/07/2026", "15-Jul-26" -> "15/07/2026"
    """
    if not raw:
        return []
    clean = str(raw).strip()
    clean = clean.replace(".", "/").replace("-", "/")

    # Extract tokens split by slash
    parts = clean.split("/")
    if len(parts) != 3:
        return []

    candidates: list[tuple[str, float]] = []

    # Map characters in numeric day, month, year
    def _to_digits(s: str) -> tuple[str, int]:
        d = []
        c = 0
        for ch in s:
            if ch.isdigit():
                d.append(ch)
            elif ch in LETTER_TO_DIGIT:
                d.append(LETTER_TO_DIGIT[ch])
                c += 1
            else:
                return "", 99
        return "".join(d), c

    p0, c0 = _to_digits(parts[0])
    p1, c1 = _to_digits(parts[1])
    p2, c2 = _to_digits(parts[2])

    if p0 and p1 and p2:
        # Standardize 2-digit year
        if len(p2) == 2:
            p2 = f"20{p2}"
        if len(p0) == 1:
            p0 = f"0{p0}"
        if len(p1) == 1:
            p1 = f"0{p1}"

        cand = f"{p0}/{p1}/{p2}"
        # Validate date
        for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(cand, fmt)
                std_date = dt.strftime("%d/%m/%Y")
                total_changes = c0 + c1 + c2
                conf = max(0.60, 0.95 - (0.08 * total_changes))
                candidates.append((std_date, round(conf, 2)))
                break
            except ValueError:
                pass

    return candidates


def correct_phone(raw: str) -> list[tuple[str, float]]:
    """Corrects 10-digit Indian mobile numbers (starts with 6-9)."""
    if not raw:
        return []
    clean = re.sub(r"[\s+-]", "", str(raw))
    if clean.startswith("91") and len(clean) == 12:
        clean = clean[2:]

    sub_digits = []
    changes = 0
    for ch in clean:
        if ch.isdigit():
            sub_digits.append(ch)
        elif ch in LETTER_TO_DIGIT:
            sub_digits.append(LETTER_TO_DIGIT[ch])
            changes += 1
        else:
            return []

    cand = "".join(sub_digits)
    if len(cand) == 10 and cand[0] in "6789":
        conf = max(0.60, 0.95 - (0.10 * changes))
        return [(cand, round(conf, 2))]
    return []
