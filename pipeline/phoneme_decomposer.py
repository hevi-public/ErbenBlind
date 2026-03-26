"""Decompose Hungarian words into IPA phoneme sequences.

Applies longest-match-first orthographic rules from hungarian_orthography.json,
handles gemination (doubled consonants), and tags each phoneme as consonant or vowel.
"""

import unicodedata
from typing import Literal, TypedDict

from pipeline.config_loader import load_config


class PhonemeToken(TypedDict):
    ipa: str
    position: int
    type: Literal["consonant", "vowel"]
    hungarian_spelling: str
    is_geminate: bool


def _build_orthography_rules() -> list[tuple[str, str, bool]]:
    """Build orthography rules sorted by decreasing length for longest-match-first.

    Includes doubled-digraph and doubled-trigraph rules at highest priority.

    Returns:
        List of (hungarian_spelling, ipa_symbol, is_geminate) tuples,
        sorted longest first, then geminates before non-geminates at same length.
    """
    ortho = load_config("hungarian_orthography.json")

    rules: list[tuple[str, str, bool]] = []

    # Build doubled forms from digraphs and trigraphs.
    # Hungarian gemination of digraph "xy" is spelled "xxy" (first letter doubled + digraph).
    # For trigraph "xyz", doubled is "xxyz".
    for section_key in ("trigraphs", "digraphs"):
        section = ortho.get(section_key, {})
        for spelling, ipa in section.items():
            doubled = spelling[0] + spelling
            rules.append((doubled, ipa, True))

    # Regular rules (not geminate)
    for section_key in ("trigraphs", "digraphs", "single_consonants", "vowels"):
        section = ortho.get(section_key, {})
        for spelling, ipa in section.items():
            rules.append((spelling, ipa, False))

    # Doubled single consonants (e.g., "tt" → "t", geminate)
    for spelling, ipa in ortho.get("single_consonants", {}).items():
        doubled = spelling + spelling
        rules.append((doubled, ipa, True))

    # Sort: longest first. At same length, geminates first (so "ssz" beats "ss"+"z").
    rules.sort(key=lambda r: (-len(r[0]), not r[2]))

    return rules


def _build_phoneme_type_lookup() -> dict[str, Literal["consonant", "vowel"]]:
    """Build a lookup from IPA symbol to 'consonant' or 'vowel'."""
    features = load_config("phoneme_features.json")
    lookup: dict[str, Literal["consonant", "vowel"]] = {}
    for ipa, entry in features.get("consonants", {}).items():
        lookup[ipa] = "consonant"
    for ipa, entry in features.get("vowels", {}).items():
        lookup[ipa] = "vowel"
    return lookup


def decompose_word(word: str) -> list[PhonemeToken]:
    """Decompose a Hungarian word into an ordered sequence of IPA phonemes.

    Applies longest-match-first decomposition using hungarian_orthography.json.
    Geminated (doubled) consonants are represented as a single phoneme token
    with is_geminate=True, since Erben activation is the same regardless of length.

    Args:
        word: Hungarian word string (e.g., "rút", "szállás", "gyöngy").

    Returns:
        Ordered list of PhonemeToken dicts with IPA, position, and type info.

    Raises:
        ValueError: If any character in the word cannot be matched.
    """
    rules = _build_orthography_rules()
    type_lookup = _build_phoneme_type_lookup()

    # NFC-normalize to ensure precomposed forms match orthography keys
    normalized = unicodedata.normalize("NFC", word.lower())

    raw_tokens: list[tuple[str, str, bool]] = []  # (ipa, spelling, is_geminate)
    i = 0

    while i < len(normalized):
        matched = False
        for spelling, ipa, is_geminate in rules:
            if normalized[i:i + len(spelling)] == spelling:
                raw_tokens.append((ipa, spelling, is_geminate))
                i += len(spelling)
                matched = True
                break

        if not matched:
            raise ValueError(
                f"Unrecognized character at position {i}: '{normalized[i]}' "
                f"in word '{word}'"
            )

    # Collapse consecutive identical IPA symbols (non-geminate duplicates from
    # regular matching that weren't caught by the doubled-form rules)
    collapsed: list[tuple[str, str, bool]] = []
    for ipa, spelling, is_gem in raw_tokens:
        if collapsed and collapsed[-1][0] == ipa and not collapsed[-1][2]:
            # Merge: mark as geminate, keep the first spelling
            prev_ipa, prev_spelling, _ = collapsed[-1]
            collapsed[-1] = (prev_ipa, prev_spelling, True)
        else:
            collapsed.append((ipa, spelling, is_gem))

    # Build final tokens with positions and type tags
    tokens: list[PhonemeToken] = []
    for idx, (ipa, spelling, is_gem) in enumerate(collapsed):
        phoneme_type = type_lookup.get(ipa)
        if phoneme_type is None:
            raise ValueError(
                f"IPA symbol '{ipa}' (from spelling '{spelling}') not found "
                f"in phoneme_features.json"
            )
        tokens.append(PhonemeToken(
            ipa=ipa,
            position=idx + 1,
            type=phoneme_type,
            hungarian_spelling=spelling,
            is_geminate=is_gem,
        ))

    return tokens


if __name__ == "__main__":
    test_words = ["rút", "szállás", "gyöngy", "kutya", "pöttynyes"]

    for word in test_words:
        try:
            phonemes = decompose_word(word)
            ipa_seq = [p["ipa"] for p in phonemes]
            types = [p["type"][0] for p in phonemes]  # c/v shorthand
            gems = ["G" if p["is_geminate"] else "." for p in phonemes]
            print(f"{word:12s} → {ipa_seq}")
            print(f"{'':12s}   types: {types}")
            print(f"{'':12s}   gemin: {gems}")
            print()
        except ValueError as e:
            print(f"{word:12s} → ERROR: {e}")
            print()
