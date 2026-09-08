"""Noise Injection Module for Simulating Dirty Customer Complaint Data.

Simulates realistic noisy customer text: spelling mistakes, repeated characters,
emojis, omitted punctuation, informal phrasing. Follows ADR Phase 1.
"""

import random
from typing import List

# Common noise elements
EMOJIS_COMPLAINT = ["😡", "👎", "😠", "💔", "🤦‍♂️", "😤", "⚠️", "❌"]
EMOJIS_POSITIVE = ["👍", "😊", "✨", "💯", "👌", "❤️"]

ARABIC_TYPOS = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ة": "ه",
    "ي": "ى",
    "ى": "ي",
}

ENGLISH_TYPOS = {
    "the": "teh",
    "bottle": "botle",
    "broken": "brokn",
    "damage": "damge",
    "replacement": "replacment",
    "return": "retrun",
    "product": "prduct",
    "quality": "qualty",
    "dirty": "drty",
}


def inject_noise(
    text: str,
    is_complaint: bool = True,
    noise_rate: float = 0.3,
    rng: random.Random = None
) -> str:
    """Inject controlled realistic noise into synthetic text.

    Args:
        text: Input clean text
        is_complaint: Whether this is a complaint (affects emoji selection)
        noise_rate: Probability of applying each type of noise (0.0 to 1.0)
        rng: Random number generator instance (for reproducibility)

    Returns:
        Noisy text string
    """
    if rng is None:
        rng = random.Random(42)

    words = text.split()
    noisy_words = []

    for word in words:
        # Typo injection
        if rng.random() < noise_rate:
            # Arabic typo
            for k, v in ARABIC_TYPOS.items():
                if k in word and rng.random() < 0.5:
                    word = word.replace(k, v)
                    break
            # English typo
            lower_word = word.lower()
            if lower_word in ENGLISH_TYPOS and rng.random() < 0.7:
                word = ENGLISH_TYPOS[lower_word]

        # Character repetition (e.g. "جداااا" or "sooooo")
        if rng.random() < (noise_rate * 0.4) and len(word) > 2:
            idx = rng.randint(0, len(word) - 1)
            char = word[idx]
            word = word[:idx] + (char * rng.randint(2, 4)) + word[idx + 1:]

        noisy_words.append(word)

    result = " ".join(noisy_words)

    # Missing punctuation (remove commas, dots)
    if rng.random() < (noise_rate * 0.6):
        result = result.replace(",", "").replace(".", "").replace("،", "")

    # Emoji injection at the end
    if rng.random() < (noise_rate * 0.8):
        emojis = EMOJIS_COMPLAINT if is_complaint else EMOJIS_POSITIVE
        emoji_count = rng.randint(1, 2)
        chosen_emojis = "".join(rng.choices(emojis, k=emoji_count))
        result = f"{result} {chosen_emojis}"

    return result.strip()
