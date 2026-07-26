import unicodedata
import unittest

from minirag.utils import clean_str, clean_text, compute_mdhash_id

# Same text, both valid Unicode encodings of it. They render identically.
VI_TEXT = "Nguyễn Văn An đi đánh cầu lông ở Nhà thi đấu Phú Thọ."
VI_ENTITY = "PHÚ THỌ"


def nfc(s):
    return unicodedata.normalize("NFC", s)


def nfd(s):
    return unicodedata.normalize("NFD", s)


class TestUnicodeNormalization(unittest.TestCase):
    def test_the_two_forms_really_do_differ(self):
        """Guard the premise: without normalisation these are distinct strings."""
        self.assertNotEqual(nfc(VI_TEXT), nfd(VI_TEXT))
        self.assertNotEqual(len(nfc(VI_TEXT).encode()), len(nfd(VI_TEXT).encode()))

    def test_clean_text_collapses_both_forms(self):
        self.assertEqual(clean_text(nfc(VI_TEXT)), clean_text(nfd(VI_TEXT)))

    def test_clean_str_collapses_both_forms(self):
        self.assertEqual(clean_str(nfc(VI_ENTITY)), clean_str(nfd(VI_ENTITY)))

    def test_document_id_is_stable_across_forms(self):
        """doc_id drives dedup; an unstable one re-indexes the same document."""
        a = compute_mdhash_id(clean_text(nfc(VI_TEXT)), prefix="doc-")
        b = compute_mdhash_id(clean_text(nfd(VI_TEXT)), prefix="doc-")
        self.assertEqual(a, b)

    def test_entity_vector_id_is_stable_across_forms(self):
        a = compute_mdhash_id(clean_str(nfc(VI_ENTITY)), prefix="ent-")
        b = compute_mdhash_id(clean_str(nfd(VI_ENTITY)), prefix="ent-")
        self.assertEqual(a, b)

    def test_output_is_nfc(self):
        for form in (nfc, nfd):
            self.assertEqual(clean_text(form(VI_TEXT)), nfc(clean_text(form(VI_TEXT))))
            self.assertEqual(clean_str(form(VI_ENTITY)), nfc(clean_str(form(VI_ENTITY))))

    def test_upper_before_clean_str_still_converges(self):
        """operate.py upper-cases entity names before cleaning them."""
        self.assertEqual(
            clean_str(nfd("Nguyễn Văn An").upper()),
            clean_str(nfc("Nguyễn Văn An").upper()),
        )

    def test_existing_behaviour_is_preserved(self):
        self.assertEqual(clean_text("  hello\x00 world  "), "hello world")
        self.assertEqual(clean_str("  &amp;text\x07  "), "&text")
        self.assertEqual(clean_str(123), 123)  # non-str passes through untouched

    def test_ascii_is_untouched(self):
        """The change must be a no-op for text that has only one encoding."""
        for s in ("Wolfgang Schulz", "the badminton court", "LIHUA"):
            self.assertEqual(clean_str(s), s)
            self.assertEqual(clean_text(s), s)


if __name__ == "__main__":
    unittest.main()
