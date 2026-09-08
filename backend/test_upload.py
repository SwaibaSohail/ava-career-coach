import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from fastapi import HTTPException

import config
import main


def test_ensure_pdf_accepts_valid_pdf_bytes():
    main._ensure_pdf(b"%PDF-1.7\nsome pdf content here")  # must not raise


def test_ensure_pdf_rejects_non_pdf():
    with pytest.raises(HTTPException) as exc:
        main._ensure_pdf(b"just plain text, definitely not a pdf")
    assert exc.value.status_code == 400


def test_ensure_pdf_rejects_empty():
    with pytest.raises(HTTPException) as exc:
        main._ensure_pdf(b"")
    assert exc.value.status_code == 400


def test_ensure_pdf_rejects_oversized(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(HTTPException) as exc:
        main._ensure_pdf(b"%PDF-1.7" + b"x" * 100)
    assert exc.value.status_code == 400
