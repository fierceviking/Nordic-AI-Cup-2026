"""Text normalisation, tokenisation and the number/unit reading the
hard-negative questions turn on.

A hard negative is usually the true statement with one detail swapped — the
dose, the number of weeks, the side of the body — so numbers and units are
first-class here, not incidental tokens.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

STOPWORDS: Set[str] = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
    'do', 'does', 'did', 'done', 'have', 'has', 'had', 'having', 'will',
    'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',
    'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by', 'from', 'as', 'that',
    'this', 'these', 'those', 'there', 'here', 'it', 'its', 'and', 'or', 'but',
    'if', 'then', 'than', 'so', 'about', 'any', 'some', 'all', 'you', 'your',
    'he', 'she', 'they', 'them', 'his', 'her', 'their', 'we', 'us', 'our',
    'i', 'me', 'my', 'what', 'which', 'who', 'whom', 'when', 'where', 'how',
    'right', 'okay', 'yes', 'no', 'not', 'up', 'out', 'over', 'into', 'been',
    'also', 'just', 'very', 'more', 'most', 'other', 'such', 'own', 'same',
    'too', 'only', 'now', 'well', 'get', 'got', 'go', 'going',
}

# Words that carry no information in a question but plenty of tokens.
QUESTION_FRAME = {
    'patient', 'doctor', 'conversation', 'discussion', 'discussed',
    'mention', 'mentioned', 'mentions', 'talk', 'talks', 'talked', 'said',
    'say', 'says', 'tell', 'told', 'report', 'reported', 'reports',
    'consultation', 'visit', 'appointment', 'anything', 'something',
}

NUMBER_WORDS: Dict[str, float] = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
    'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
    'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100,
    'thousand': 1000, 'million': 1_000_000,
    'half': 0.5, 'quarter': 0.25,
    'once': 1, 'twice': 2, 'thrice': 3, 'single': 1, 'double': 2,
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'daily': 1,
}

# Unit spellings collapsed onto one canonical form, so "milligrams", "mg" and
# "milligram" compare equal.
UNIT_ALIASES: Dict[str, str] = {
    'mg': 'mg', 'milligram': 'mg', 'milligrams': 'mg', 'milligramme': 'mg',
    'mcg': 'ug', 'microgram': 'ug', 'micrograms': 'ug', 'ug': 'ug',
    'g': 'g', 'gram': 'g', 'grams': 'g', 'gramme': 'g', 'grammes': 'g',
    'kg': 'kg', 'kilo': 'kg', 'kilos': 'kg', 'kilogram': 'kg',
    'kilograms': 'kg',
    'ml': 'ml', 'millilitre': 'ml', 'millilitres': 'ml', 'milliliter': 'ml',
    'milliliters': 'ml',
    'l': 'l', 'litre': 'l', 'litres': 'l', 'liter': 'l', 'liters': 'l',
    'iu': 'iu', 'units': 'iu', 'unit': 'iu',
    'day': 'day', 'days': 'day', 'daily': 'day',
    'week': 'week', 'weeks': 'week', 'weekly': 'week',
    'month': 'month', 'months': 'month', 'monthly': 'month',
    'year': 'year', 'years': 'year', 'yearly': 'year',
    'hour': 'hour', 'hours': 'hour', 'hourly': 'hour',
    'minute': 'minute', 'minutes': 'minute',
    'tablet': 'tablet', 'tablets': 'tablet', 'pill': 'tablet',
    'pills': 'tablet', 'capsule': 'tablet', 'capsules': 'tablet',
    'time': 'times', 'times': 'times',
    'mmol': 'mmol', 'mmhg': 'mmhg', 'percent': 'pct', '%': 'pct',
    'degrees': 'deg', 'degree': 'deg',
    'mmol/l': 'mmol', 'mg/dl': 'mgdl',
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./%][a-z0-9]+)*")
_NUMBER_RE = re.compile(r'\d+(?:[.,]\d+)?')

# Pairs that flip the meaning of an otherwise identical sentence. Used by the
# rule layer to spot a hard negative that swapped a polarity word.
ANTONYMS: List[Tuple[str, str]] = [
    ('increase', 'decrease'), ('increase', 'reduce'), ('increase', 'lower'),
    ('increased', 'reduced'), ('higher', 'lower'), ('more', 'less'),
    ('start', 'stop'), ('started', 'stopped'), ('start', 'continue'),
    ('begin', 'end'), ('continue', 'discontinue'), ('continue', 'stop'),
    ('normal', 'abnormal'), ('normal', 'elevated'), ('normal', 'raised'),
    ('present', 'absent'), ('before', 'after'), ('left', 'right'),
    ('improve', 'worsen'), ('improved', 'worsened'), ('better', 'worse'),
    ('stable', 'unstable'), ('changed', 'unchanged'), ('with', 'without'),
    ('positive', 'negative'), ('high', 'low'), ('up', 'down'),
    ('morning', 'evening'), ('empty', 'full'), ('above', 'below'),
]


def normalise(text: str) -> str:
    text = text.lower()
    text = text.replace('-', ' ').replace('/', ' / ')
    text = re.sub(r"[’']s\b", '', text)
    text = re.sub(r"[^a-z0-9.%/ ]+", ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(normalise(text))


def stem(token: str) -> str:
    """A deliberately small suffix stripper; no NLTK dependency."""
    for suffix in ('ations', 'ation', 'ings', 'ing', 'edly', 'ies', 'ied',
                   'ers', 'er', 'ed', 'es', 's', 'ly'):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            base = token[: -len(suffix)]
            if suffix in ('ies', 'ied'):
                base += 'y'
            return base
    return token


def content_tokens(text: str, drop_frame: bool = False) -> List[str]:
    tokens = tokenize(text)
    drop = STOPWORDS | (QUESTION_FRAME if drop_frame else set())
    return [t for t in tokens if t not in drop and len(t) > 1 or t.isdigit()]


def content_stems(text: str, drop_frame: bool = False) -> List[str]:
    return [stem(t) for t in content_tokens(text, drop_frame)]


# --------------------------------------------------------------------------- #
# Numbers and units
# --------------------------------------------------------------------------- #

def _word_number_sequence(tokens: List[str], index: int) -> Tuple[float, int]:
    """Read a spelled-out number starting at ``index``. Returns (value, length)."""
    total = 0.0
    current = 0.0
    length = 0
    seen = False

    while index + length < len(tokens):
        token = tokens[index + length]
        if token == 'and' and seen:
            length += 1
            continue
        if token not in NUMBER_WORDS:
            break

        value = NUMBER_WORDS[token]
        if value == 100:
            current = (current or 1) * 100
        elif value in (1000, 1_000_000):
            total += (current or 1) * value
            current = 0.0
        else:
            current += value
        seen = True
        length += 1

    if not seen:
        return 0.0, 0

    # "and" eaten at the tail is not part of the number.
    while length and tokens[index + length - 1] == 'and':
        length -= 1

    return total + current, length


def extract_quantities(text: str) -> Set[Tuple[float, str]]:
    """Every (value, canonical-unit) pair the text states.

    ``100 mg`` and ``a hundred milligrams`` both come back as ``(100.0, 'mg')``.
    A number with no unit nearby is kept as ``(value, '')`` so ``two weeks``
    against ``six weeks`` still differs even when the unit is missed.
    """
    tokens = tokenize(text)
    quantities: Set[Tuple[float, str]] = set()

    index = 0
    while index < len(tokens):
        token = tokens[index]
        value = None
        length = 1

        if _NUMBER_RE.fullmatch(token.replace(',', '.')):
            try:
                value = float(token.replace(',', '.'))
            except ValueError:
                value = None
        elif token in NUMBER_WORDS:
            value, length = _word_number_sequence(tokens, index)
            if length == 0:
                value = None

        if value is not None:
            unit = ''
            for offset in (length, length + 1):
                if index + offset < len(tokens):
                    candidate = UNIT_ALIASES.get(tokens[index + offset])
                    if candidate:
                        unit = candidate
                        break
            quantities.add((float(value), unit))
            index += max(length, 1)
            continue

        index += 1

    return quantities


def numbers_only(quantities: Set[Tuple[float, str]]) -> Set[float]:
    return {value for value, _ in quantities}


def quantity_conflict(
    question_quantities: Set[Tuple[float, str]],
    evidence_quantities: Set[Tuple[float, str]],
) -> bool:
    """True when the question states a quantity the evidence contradicts.

    Same unit, different value, and the value is nowhere in the evidence: that
    is the classic hard negative — ``200 mg daily`` read off a passage that
    says ``100 mg``.
    """
    if not question_quantities or not evidence_quantities:
        return False

    evidence_values = numbers_only(evidence_quantities)

    for value, unit in question_quantities:
        if value in evidence_values:
            continue
        if unit and any(unit == other_unit for _, other_unit in evidence_quantities):
            return True
        if not unit and evidence_values:
            return True

    return False


def antonym_conflict(question: str, evidence: str) -> bool:
    """True when the question uses the opposite of a word the evidence uses."""
    question_stems = set(content_stems(question))
    evidence_stems = set(content_stems(evidence))

    for left, right in ANTONYMS:
        left_stem, right_stem = stem(left), stem(right)
        if left_stem in question_stems and right_stem in evidence_stems \
                and left_stem not in evidence_stems:
            return True
        if right_stem in question_stems and left_stem in evidence_stems \
                and right_stem not in evidence_stems:
            return True

    return False


def declarative(question: str) -> str:
    """Turn a yes/no question into the statement it is asking about.

    ``Should the daily dose be 100 mg?`` becomes ``the daily dose should be
    100 mg``. Crude, but an NLI model wants a hypothesis, not a question, and
    the questions here are not uniformly interrogative anyway.
    """
    text = question.strip().rstrip('?').strip()

    # Tag questions: "The lipid profile came back normal, didn't it" -> drop tag.
    text = re.sub(
        r',\s*(is|isn\'t|are|aren\'t|was|wasn\'t|were|weren\'t|do|don\'t|'
        r'does|doesn\'t|did|didn\'t|has|hasn\'t|have|haven\'t|will|won\'t|'
        r'can\'t|right|correct)\s*(it|he|she|they|that|this)?\s*$',
        '', text, flags=re.IGNORECASE,
    ).strip()

    words = text.split()
    if not words:
        return text

    leading = words[0].lower()
    auxiliaries = {
        'is', 'are', 'was', 'were', 'do', 'does', 'did', 'has', 'have', 'had',
        'will', 'would', 'should', 'shall', 'can', 'could', 'may', 'might',
        'must',
    }

    if leading in auxiliaries and len(words) > 2:
        # Dropping the fronted auxiliary keeps the word order intact. Moving it
        # instead needs to know where the subject ends, and guessing at a fixed
        # offset turned "Was the patient listened to" into "The patient listened
        # was to" for about half the questions.
        statement = ' '.join(words[1:])
        return statement[0].upper() + statement[1:] + '.'

    return text + '.'
