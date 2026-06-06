"""
Unit tests for input validation + the SQL safety guard (app/utils.py).

The README's Design Decisions table claims dangerous SQL (DROP/DELETE/TRUNCATE/
ALTER) is blocked before execution — these tests pin that behavior.
"""

import pytest

from app.utils import FileValidator, QueryValidator, ValidationError

# ── SQL safety guard ────────────────────────────────────────────────────────

@pytest.mark.parametrize("sql", [
    "DROP TABLE defects;",
    "delete from capa_log where id = 1",
    "TRUNCATE ncr",
    "ALTER TABLE suppliers ADD COLUMN x int",
    "INSERT INTO defects VALUES (1)",
    "UPDATE defects SET severity = 1",
    "CREATE TABLE evil (id int)",
])
def test_dangerous_sql_is_flagged(sql):
    assert QueryValidator.check_dangerous_sql(sql) is True


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM capa_log WHERE status = 'open'",
    "SELECT part_number, cpk FROM inspection_results ORDER BY cpk ASC",
    "select * from suppliers where quality_rating < 80",
])
def test_safe_select_is_allowed(sql):
    assert QueryValidator.check_dangerous_sql(sql) is False


def test_guard_is_case_insensitive():
    assert QueryValidator.check_dangerous_sql("dRoP tAbLe defects") is True


# ── Question validation ─────────────────────────────────────────────────────

def test_valid_question_is_trimmed():
    assert QueryValidator.validate_question("  how many NCRs?  ") == "how many NCRs?"


def test_empty_question_rejected():
    with pytest.raises(ValidationError):
        QueryValidator.validate_question("   ")


def test_empty_question_allowed_when_flagged():
    assert QueryValidator.validate_question("   ", allow_empty=True) == ""


def test_too_short_question_rejected():
    with pytest.raises(ValidationError):
        QueryValidator.validate_question("hi")


def test_too_long_question_rejected():
    with pytest.raises(ValidationError):
        QueryValidator.validate_question("x" * 1001)


# ── top_k validation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [1, 5, 10])
def test_valid_top_k(value):
    assert QueryValidator.validate_top_k(value) == value


@pytest.mark.parametrize("value", [0, -1, 11])
def test_out_of_range_top_k_rejected(value):
    with pytest.raises(ValidationError):
        QueryValidator.validate_top_k(value)


def test_non_int_top_k_rejected():
    with pytest.raises(ValidationError):
        QueryValidator.validate_top_k("3")


# ── SQL display sanitization ────────────────────────────────────────────────

def test_sanitize_strips_comments_and_normalizes_whitespace():
    raw = "SELECT *  FROM defects -- secret\n/* block */ WHERE id = 1"
    cleaned = QueryValidator.sanitize_sql_for_display(raw)
    assert "--" not in cleaned and "/*" not in cleaned
    assert "  " not in cleaned


# ── File validation ─────────────────────────────────────────────────────────

class _FakeUpload:
    def __init__(self, filename, size=None):
        self.filename = filename
        if size is not None:
            self.size = size


def test_allowed_extension_passes():
    FileValidator.validate_file(_FakeUpload("PFMEA.pdf"))  # no raise


def test_disallowed_extension_rejected():
    with pytest.raises(ValidationError):
        FileValidator.validate_file(_FakeUpload("malware.exe"))


def test_oversize_file_rejected():
    with pytest.raises(ValidationError):
        FileValidator.validate_file(_FakeUpload("big.pdf", size=FileValidator.MAX_FILE_SIZE + 1))


def test_missing_filename_rejected():
    with pytest.raises(ValidationError):
        FileValidator.validate_file(_FakeUpload(""))
