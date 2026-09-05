import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guardrails import GuardResult, truncate, generate_guarded_reply, guard_incoming


def test_truncate_caps_length():
    assert len(truncate("a" * 600)) == 500


def test_truncate_strips_control_chars_but_keeps_newlines():
    assert truncate("hi\x00\x07there") == "hithere"
    assert truncate("line1\nline2") == "line1\nline2"


def test_generate_guarded_reply_is_nonempty_for_each_category():
    for category in ("injection", "abuse", "spam", "harmful"):
        assert generate_guarded_reply(category).strip()


def test_generate_guarded_reply_never_empty_for_unknown_category():
    assert generate_guarded_reply(None).strip()


def test_guard_incoming_allows_a_clean_message():
    result = guard_incoming("Please tailor my CV for a frontend role")
    assert isinstance(result, GuardResult)
    assert result.allowed is True
    assert result.category is None
    assert result.safe_reply is None
    assert result.cleaned_message == "Please tailor my CV for a frontend role"
