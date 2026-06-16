"""
tests/test_utils.py
Unit tests for RecruitRadar utility functions.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    clean_text,
    extract_keywords,
    find_keyword_gaps,
    format_score_label,
    truncate_text
)


class TestCleanText:

    def test_removes_extra_whitespace(self):
        text = "hello   world"
        assert "  " not in clean_text(text)

    def test_removes_extra_newlines(self):
        text = "line1\n\n\n\nline2"
        assert "\n\n\n" not in clean_text(text)

    def test_returns_string(self):
        assert isinstance(clean_text("test"), str)


class TestExtractKeywords:

    def test_returns_list(self):
        assert isinstance(extract_keywords("Python developer"), list)

    def test_removes_stop_words(self):
        keywords = extract_keywords("the quick brown fox")
        assert "the" not in keywords

    def test_filters_short_words(self):
        keywords = extract_keywords("a an the is")
        assert all(len(k) > 2 for k in keywords)

    def test_lowercases_keywords(self):
        keywords = extract_keywords("Python JAVA JavaScript")
        assert all(k == k.lower() for k in keywords)


class TestFindKeywordGaps:

    def test_returns_dict(self):
        result = find_keyword_gaps("Python developer", "Python engineer")
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = find_keyword_gaps("Python developer", "Python engineer")
        assert "matched_keywords" in result
        assert "missing_keywords" in result
        assert "match_percentage" in result

    def test_perfect_match(self):
        result = find_keyword_gaps("python developer", "python developer")
        assert result["match_percentage"] == 100.0

    def test_no_match(self):
        result = find_keyword_gaps("java backend", "python machine learning")
        assert result["match_percentage"] < 50


class TestFormatScoreLabel:

    def test_strong_match(self):
        label = format_score_label(85)
        assert "Strong" in label
        assert "85" in label

    def test_moderate_match(self):
        label = format_score_label(60)
        assert "Moderate" in label

    def test_weak_match(self):
        label = format_score_label(30)
        assert "Weak" in label


class TestTruncateText:

    def test_short_text_unchanged(self):
        text = "short text"
        assert truncate_text(text, 100) == text

    def test_long_text_truncated(self):
        text = "word " * 200
        result = truncate_text(text, 50)
        assert len(result) <= 54

    def test_ends_with_ellipsis(self):
        text = "word " * 200
        result = truncate_text(text, 50)
        assert result.endswith("...")