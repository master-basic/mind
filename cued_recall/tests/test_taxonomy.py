"""The fixed tag vocabulary -- the system's only non-vector representation.

Validation is the boundary between a 1.5B model's free text and a controlled
vocabulary; if it leaks, tag filtering silently stops being comparable across
instances.
"""

from cued_recall.taxonomy import (
    TAXONOMY,
    TAXONOMY_GROUPS,
    validate_gist,
    validate_tags,
)


class TestTaxonomy:
    def test_no_tag_appears_in_two_groups(self):
        assert len(TAXONOMY) == len(set(TAXONOMY))

    def test_flat_list_is_the_union_of_the_groups(self):
        assert set(TAXONOMY) == {t for g in TAXONOMY_GROUPS.values() for t in g}


class TestValidateTags:
    def test_keeps_known_tags(self):
        assert validate_tags(["python", "dns"]) == ["python", "dns"]

    def test_drops_invented_tags(self):
        # The failure this prevents: one instance tagging "network-stuff" where
        # another tagged "networking".
        assert validate_tags(["network-stuff", "python"]) == ["python"]

    def test_normalises_case_and_whitespace(self):
        assert validate_tags(["  PYTHON  "]) == ["python"]

    def test_dedupes(self):
        assert validate_tags(["python", "Python", "python"]) == ["python"]

    def test_caps_at_max_tags(self):
        assert len(validate_tags(["python", "dns", "docker", "rust"])) == 3
        assert len(validate_tags(["python", "dns", "docker", "rust"],
                                 max_tags=2)) == 2

    def test_handles_none_and_junk(self):
        assert validate_tags(None) == []
        assert validate_tags([]) == []
        assert validate_tags([None, 1, {"a": 1}]) == []


class TestValidateGist:
    def test_short_gist_untouched(self):
        assert validate_gist("a short gist") == "a short gist"

    def test_strips(self):
        assert validate_gist("  spaced  ") == "spaced"

    def test_long_gist_is_ellipsised_within_the_limit(self):
        g = validate_gist("x" * 100, max_chars=40)
        assert len(g) <= 40
        assert g.endswith("…")

    def test_none_becomes_empty(self):
        assert validate_gist(None) == ""
