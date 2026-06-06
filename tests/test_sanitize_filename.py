"""Tests for upload filename sanitization."""

import pytest

from app.utils import FileValidator, ValidationError


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("PFMEA.pdf", "PFMEA.pdf"),
        ("../../etc/passwd", "passwd"),
        ("nested/report.docx", "report.docx"),
    ],
)
def test_sanitize_filename_strips_path_components(raw, expected):
    assert FileValidator.sanitize_filename(raw) == expected


@pytest.mark.parametrize("raw", ["", "..", ".", "bad name.pdf", "file;drop.pdf"])
def test_sanitize_filename_rejects_invalid(raw):
    with pytest.raises(ValidationError):
        FileValidator.sanitize_filename(raw)
