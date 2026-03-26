"""Tests for the phoneme decomposer module."""

import unittest

from pipeline.phoneme_decomposer import decompose_word


class TestBasicDecomposition(unittest.TestCase):
    """Core decomposition of simple Hungarian words."""

    def test_rut(self):
        """'rút' → r, uː, t — the canonical test word from CLAUDE.md."""
        tokens = decompose_word("rút")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["r", "uː", "t"])

    def test_viz(self):
        """'víz' → v, iː, z — simple CVC with long vowel."""
        tokens = decompose_word("víz")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["v", "iː", "z"])

    def test_tuz(self):
        """'tűz' → t, yː, z — front rounded long vowel."""
        tokens = decompose_word("tűz")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["t", "yː", "z"])

    def test_kor(self):
        """'kör' → k, ø, r — front rounded short vowel."""
        tokens = decompose_word("kör")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["k", "ø", "r"])


class TestDigraphsAndTrigraphs(unittest.TestCase):
    """Digraph and trigraph handling with longest-match-first."""

    def test_sz_digraph(self):
        """'szó' → s, oː — 'sz' is a digraph mapping to /s/."""
        tokens = decompose_word("szó")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["s", "oː"])

    def test_cs_digraph(self):
        """'cső' → tʃ, øː — 'cs' digraph."""
        tokens = decompose_word("cső")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["tʃ", "øː"])

    def test_dzs_trigraph(self):
        """'dzsem' → dʒ, ɛ, m — 'dzs' trigraph beats 'dz' digraph + 's'."""
        tokens = decompose_word("dzsem")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["dʒ", "ɛ", "m"])

    def test_gy_palatal_stop(self):
        """'gyöngy' → ɟ, ø, n, ɟ — tests palatal stop from gy digraph."""
        tokens = decompose_word("gyöngy")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["ɟ", "ø", "n", "ɟ"])

    def test_ty_palatal_stop(self):
        """'kutya' → k, u, c, ɒ — 'ty' maps to voiceless palatal stop /c/."""
        tokens = decompose_word("kutya")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["k", "u", "c", "ɒ"])

    def test_ny_digraph(self):
        """'nyár' → ɲ, aː, r — 'ny' palatal nasal."""
        tokens = decompose_word("nyár")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["ɲ", "aː", "r"])

    def test_ly_digraph(self):
        """'lyuk' → j, u, k — 'ly' maps to /j/ in modern Hungarian."""
        tokens = decompose_word("lyuk")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["j", "u", "k"])


class TestGemination(unittest.TestCase):
    """Doubled consonant handling — geminates collapse to single phoneme."""

    def test_doubled_single_consonant(self):
        """'szállás' — doubled 'l' collapses to single geminate /l/."""
        tokens = decompose_word("szállás")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["s", "aː", "l", "aː", "ʃ"])
        # The 'l' should be marked geminate
        l_token = [t for t in tokens if t["ipa"] == "l"][0]
        self.assertTrue(l_token["is_geminate"])

    def test_doubled_tt(self):
        """'itt' → i, t — doubled 't' collapses."""
        tokens = decompose_word("itt")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["i", "t"])
        t_token = [t for t in tokens if t["ipa"] == "t"][0]
        self.assertTrue(t_token["is_geminate"])

    def test_doubled_digraph_ssz(self):
        """'hosszú' → h, o, s, uː — 'ssz' is doubled 'sz', collapses to /s/."""
        tokens = decompose_word("hosszú")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["h", "o", "s", "uː"])
        s_token = [t for t in tokens if t["ipa"] == "s"][0]
        self.assertTrue(s_token["is_geminate"])

    def test_doubled_digraph_ccs(self):
        """'meccs' → m, ɛ, tʃ — 'ccs' is doubled 'cs'."""
        tokens = decompose_word("meccs")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["m", "ɛ", "tʃ"])
        cs_token = [t for t in tokens if t["ipa"] == "tʃ"][0]
        self.assertTrue(cs_token["is_geminate"])

    def test_doubled_digraph_nny(self):
        """'könny' → k, ø, ɲ — 'nny' is doubled 'ny'."""
        tokens = decompose_word("könny")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["k", "ø", "ɲ"])
        ny_token = [t for t in tokens if t["ipa"] == "ɲ"][0]
        self.assertTrue(ny_token["is_geminate"])

    def test_doubled_digraph_ggy(self):
        """'meggy' → m, ɛ, ɟ — 'ggy' is doubled 'gy'."""
        tokens = decompose_word("meggy")
        ipas = [t["ipa"] for t in tokens]
        self.assertEqual(ipas, ["m", "ɛ", "ɟ"])
        gy_token = [t for t in tokens if t["ipa"] == "ɟ"][0]
        self.assertTrue(gy_token["is_geminate"])

    def test_non_geminate_tokens(self):
        """'rút' — no geminates, all tokens should have is_geminate=False."""
        tokens = decompose_word("rút")
        for t in tokens:
            self.assertFalse(t["is_geminate"], f"{t['ipa']} should not be geminate")


class TestPositionAndType(unittest.TestCase):
    """Position numbering and consonant/vowel type tagging."""

    def test_positions_are_one_indexed(self):
        tokens = decompose_word("rút")
        positions = [t["position"] for t in tokens]
        self.assertEqual(positions, [1, 2, 3])

    def test_types_rut(self):
        tokens = decompose_word("rút")
        types = [t["type"] for t in tokens]
        self.assertEqual(types, ["consonant", "vowel", "consonant"])

    def test_types_kutya(self):
        tokens = decompose_word("kutya")
        types = [t["type"] for t in tokens]
        self.assertEqual(types, ["consonant", "vowel", "consonant", "vowel"])


class TestHungarianVowelDistinction(unittest.TestCase):
    """Hungarian 'a' /ɒ/ vs 'á' /aː/ — critical distinction."""

    def test_a_is_rounded_back(self):
        """Hungarian 'a' → /ɒ/ (rounded back), not /a/."""
        tokens = decompose_word("hal")
        self.assertEqual(tokens[1]["ipa"], "ɒ")

    def test_a_acute_is_unrounded_front(self):
        """Hungarian 'á' → /aː/ (unrounded front)."""
        tokens = decompose_word("hát")
        self.assertEqual(tokens[1]["ipa"], "aː")


class TestEdgeCases(unittest.TestCase):

    def test_uppercase_input(self):
        """Decomposition should be case-insensitive."""
        tokens_lower = decompose_word("rút")
        tokens_upper = decompose_word("RÚT")
        ipas_lower = [t["ipa"] for t in tokens_lower]
        ipas_upper = [t["ipa"] for t in tokens_upper]
        self.assertEqual(ipas_lower, ipas_upper)

    def test_single_vowel(self):
        """Single-character word should work."""
        tokens = decompose_word("ó")
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0]["ipa"], "oː")
        self.assertEqual(tokens[0]["type"], "vowel")

    def test_unrecognized_character_raises(self):
        """Characters not in the orthography should raise ValueError."""
        with self.assertRaises(ValueError):
            decompose_word("hello!")  # '!' is not in orthography

    def test_hungarian_s_is_sh(self):
        """Hungarian single 's' → /ʃ/, not /s/. 'sz' → /s/."""
        tokens = decompose_word("só")
        self.assertEqual(tokens[0]["ipa"], "ʃ")

        tokens = decompose_word("szó")
        self.assertEqual(tokens[0]["ipa"], "s")

    def test_c_is_ts(self):
        """Hungarian 'c' → /ts/ (alveolar affricate)."""
        tokens = decompose_word("cél")
        self.assertEqual(tokens[0]["ipa"], "ts")


if __name__ == "__main__":
    unittest.main()
