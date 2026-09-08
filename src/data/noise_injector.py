"""
Controlled Noise Injector for Synthetic Text.
Phase 1: Paired Dataset - Injects typos, emoji, repetition, and casing variations.
"""

import random
from typing import Optional


class NoiseInjector:
    """Injects controlled noise into text for realistic dirty data simulation."""

    # Common typographical errors (keyboard neighbor substitutions)
    TYPO_MAP = {
        'a': ['s', 'q', 'z'],
        'e': ['w', 'r', 'd'],
        'i': ['u', 'o', 'k'],
        'o': ['i', 'p', 'l'],
        'u': ['y', 'i', 'j'],
        'b': ['v', 'g', 'n'],
        'd': ['s', 'f', 'e'],
        't': ['r', 'y', 'g'],
    }

    # Emojis for dirty data simulation
    EMOJIS = ['😠', '😡', '👎', '⚠️', '💔', '😢', '😤', '📦', '💥', '🤷‍♂️']

    def __init__(self, noise_rate: float = 0.3, seed: Optional[int] = None):
        """
        Initialize noise injector.

        Args:
            noise_rate: Probability of applying noise to a sample (0.0 to 1.0)
            seed: Random seed for deterministic reproducibility
        """
        self.noise_rate = noise_rate
        self.rng = random.Random(seed)

    def inject_noise(self, text: str) -> str:
        """
        Apply controlled noise to text.

        Args:
            text: Original clean text

        Returns:
            Noisy text
        """
        if not text or self.rng.random() > self.noise_rate:
            return text

        noisy_text = text

        # 1. Typo injection (character swap or neighbor substitution)
        if self.rng.random() < 0.4 and len(noisy_text) > 5:
            pos = self.rng.randint(0, len(noisy_text) - 1)
            char = noisy_text[pos].lower()
            if char in self.TYPO_MAP:
                sub = self.rng.choice(self.TYPO_MAP[char])
                noisy_text = noisy_text[:pos] + sub + noisy_text[pos + 1:]

        # 2. Repeated character injection (e.g., "soooo", "broken!!!!")
        if self.rng.random() < 0.35 and len(noisy_text) > 4:
            pos = self.rng.randint(0, len(noisy_text) - 1)
            char = noisy_text[pos]
            if char.isalpha():
                repeat_count = self.rng.randint(2, 4)
                noisy_text = noisy_text[:pos] + (char * repeat_count) + noisy_text[pos + 1:]

        # 3. Punctuation removal or excessive punctuation
        if self.rng.random() < 0.4:
            choice = self.rng.choice(["remove", "excessive"])
            if choice == "remove":
                noisy_text = "".join(c for c in noisy_text if c not in ".,!?;:")
            else:
                punct = self.rng.choice(["!", "?", "!!", "???"])
                noisy_text = noisy_text + " " + punct

        # 4. Emoji injection
        if self.rng.random() < 0.35:
            emoji = self.rng.choice(self.EMOJIS)
            pos = self.rng.choice(["start", "end", "middle"])
            if pos == "start":
                noisy_text = f"{emoji} {noisy_text}"
            elif pos == "end":
                noisy_text = f"{noisy_text} {emoji}"
            else:
                words = noisy_text.split()
                if len(words) > 2:
                    insert_idx = self.rng.randint(1, len(words) - 1)
                    words.insert(insert_idx, emoji)
                    noisy_text = " ".join(words)

        # 5. Case variation (English only)
        if self.rng.random() < 0.3:
            choice = self.rng.choice(["upper", "lower", "mixed"])
            if choice == "upper":
                noisy_text = noisy_text.upper()
            elif choice == "lower":
                noisy_text = noisy_text.lower()

        return noisy_text
