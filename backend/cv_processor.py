"""PDF -> text: chunk a CV for retrieval, or extract it whole for analysis.

Uploaded CV text is UNTRUSTED input. Because it is embedded into Ava's prompt,
it is a prompt-injection vector, so we scrub instruction-like lines (e.g. "ignore
previous instructions", "system override") before the text is used anywhere.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
# One source of truth for the injection patterns, shared with the chat guardrails.
from guardrails import INJECTION_RE as _INJECTION_RE

_REDACTION = "[redacted: suspected injected instruction in uploaded file]"


def sanitize_cv_text(text: str) -> str:
    """Neutralize prompt-injection attempts hidden in an uploaded CV.

    A CV is data, not instructions. Any line that reads like a command aimed at
    the AI (e.g. "ignore previous instructions", "system override", "do not tell
    the user") is replaced with a redaction marker before the text is embedded or
    handed to the agent, so a tampered file can't reprogram Ava.
    """
    if not text:
        return text
    return "\n".join(
        _REDACTION if _INJECTION_RE.search(line) else line
        for line in text.splitlines()
    )


def load_and_chunk_cv(pdf_path: str) -> list:
    """Split a CV PDF into overlapping, section-aware chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    for page in pages:
        page.page_content = sanitize_cv_text(page.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # Prefer splitting on CV section headings so a section stays intact.
        separators=[
            "\nProjects",
            "\nPROJECTS",
            "\nTechnical Skills",
            "\nTECHNICAL SKILLS",
            "\nCore Skills",
            "\nSkills",
            "\nSKILLS",
            "\nWork Experience",
            "\nExperience",
            "\nEXPERIENCE",
            "\nEducation",
            "\nEDUCATION",
            "\nLicenses",
            "\nCertifications",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )
    chunks = splitter.split_documents(pages)
    return chunks


def extract_full_text(pdf_path: str) -> str:
    """Return the whole CV as one string (used for the match analysis)."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return sanitize_cv_text("\n".join(page.page_content for page in pages))
