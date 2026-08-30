"""PDF -> text: chunk a CV for retrieval, or extract it whole for analysis."""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_and_chunk_cv(pdf_path: str) -> list:
    """Split a CV PDF into overlapping, section-aware chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

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
    return "\n".join(page.page_content for page in pages)
